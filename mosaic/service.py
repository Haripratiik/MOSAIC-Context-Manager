from __future__ import annotations

"""Canonical runtime service for MOSAIC query handling, auditing, and conversation state."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from .audit import AuditTrace, FACTOR_ORDER, JsonLedgerStore, SQLiteAuditStore
from .auth import Principal, PrincipalResolver
from .evaluator import answer_query_locally, select_strategy
from .ingestor import load_index
from .ledger import ContextLedger, LedgerStore
from .retriever import hybrid_retrieve
from .signal import (
    FAILURE_SUCCESS,
    SignalThresholds,
    build_corpus_idf,
    calibrate_thresholds,
    classify_query,
)
from .types import Chunk, RetrievalIndex
from .utils import (
    DEFAULT_EMBEDDING_MODEL,
    build_context,
    cosine_similarity,
    load_json,
    roles_permit,
)

WITHIN_TURN_FACTOR_THRESHOLD = 0.9
CROSS_TURN_FACTOR_THRESHOLD = 0.9
STATUS_BY_CLASSIFICATION = {
    "SUCCESS": "success",
    "TRUE_UNKNOWN": "true_unknown",
    "PERMISSION_GAP": "permission_gap",
    "RETRIEVAL_FAILURE": "retrieval_failure",
}


@dataclass(slots=True)
class QueryRequest:
    """Input contract for the canonical MOSAIC runtime."""
    query: str
    principal_id: str | None = None
    requested_roles: list[str] = field(default_factory=list)
    strategy: str = "mosaic_full"
    candidate_k: int = 50
    token_budget: int = 4096
    conversation_id: str | None = None
    turn: int | None = None
    lam: float = 0.5
    cross_turn_lambda: float = 0.75
    return_trace: bool = False


@dataclass(slots=True)
class QueryResponse:
    """Structured runtime result returned by CLI, MCP, and UI flows."""
    status: str
    answer: str
    response: str
    classification: str
    selected_chunk_ids: list[str]
    selected_chunk_titles: list[str]
    hints: list[str]
    required_role: str | None
    audit_id: str
    conversation_id: str | None
    tokens_used: int
    remaining_budget: int | None
    timings_ms: dict[str, float]
    trace: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "answer": self.answer,
            "response": self.response,
            "classification": self.classification,
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "selected_chunk_titles": list(self.selected_chunk_titles),
            "hints": list(self.hints),
            "required_role": self.required_role,
            "audit_id": self.audit_id,
            "conversation_id": self.conversation_id,
            "tokens_used": self.tokens_used,
            "remaining_budget": self.remaining_budget,
            "timings_ms": dict(self.timings_ms),
        }
        if self.trace is not None:
            payload["trace"] = self.trace
        return payload


class MosaicService:
    """Single orchestration layer for query classification, assembly, audit, and ledger updates."""
    def __init__(
        self,
        index: RetrievalIndex | str | Path,
        *,
        audit_store: SQLiteAuditStore | None = None,
        ledger_store: LedgerStore | None = None,
        principal_resolver: PrincipalResolver | None = None,
        thresholds: SignalThresholds | None = None,
        calibration_suite_path: str | Path | None = None,
        report_path: str | Path | None = None,
    ) -> None:
        self.index = load_index(index) if isinstance(index, (str, Path)) else index
        self.audit_store = audit_store
        self.ledger_store = ledger_store
        self.principal_resolver = principal_resolver or PrincipalResolver.default()
        self.embedding_model = str(self.index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        self.chunk_lookup = {chunk.id: chunk for chunk in self.index.chunks}
        self.corpus_idf = build_corpus_idf(self.index.chunks)
        self.report_path = Path(report_path) if report_path is not None else Path(".mosaic/report.html")
        resolved_thresholds = thresholds
        if resolved_thresholds is None and calibration_suite_path:
            resolved_thresholds, _, _ = calibrate_thresholds(
                self.index.chunks,
                load_json(calibration_suite_path),
                candidate_k=50,
                embedding_model=self.embedding_model,
            )
        self.thresholds = resolved_thresholds or SignalThresholds()

    def _resolve_principal_and_roles(
        self,
        request: QueryRequest,
        principal: Principal | None,
        transport: str,
    ) -> tuple[Principal, list[str]]:
        resolved_principal = principal or self.principal_resolver.resolve_by_id(
            request.principal_id,
            transport=transport,
            default_roles=request.requested_roles,
        )
        requested_roles = [role for role in request.requested_roles if role]
        if transport == "http":
            effective_roles = self.principal_resolver.resolve_http_roles(resolved_principal, requested_roles)
        else:
            effective_roles = self.principal_resolver.resolve_stdio_roles(resolved_principal, requested_roles)
        return resolved_principal, effective_roles

    def _load_ledger(self, conversation_id: str | None, total_budget: int) -> tuple[str | None, ContextLedger | None]:
        if self.ledger_store is None:
            return conversation_id, None
        resolved_conversation_id = conversation_id or uuid4().hex
        ledger = self.ledger_store.load(resolved_conversation_id, total_budget, ContextLedger)
        return resolved_conversation_id, ledger

    def _resolve_turn(self, request_turn: int | None, ledger: ContextLedger | None) -> int:
        if request_turn is not None:
            return request_turn
        if ledger is None:
            return 1
        return len(ledger.turn_history) + 1

    def _selection_budget(self, request: QueryRequest, ledger: ContextLedger | None) -> int:
        if ledger is not None:
            return ledger.remaining_budget
        return request.token_budget

    def _prior_chunks(self, ledger: ContextLedger | None) -> list[Chunk]:
        if ledger is None:
            return []
        return [self.chunk_lookup[chunk_id] for chunk_id in ledger.selected_chunk_ids if chunk_id in self.chunk_lookup]

    def _weights_by_id(self, candidates: list[Chunk], weights: list[float]) -> dict[str, float]:
        if not candidates or len(candidates) != len(weights):
            return {}
        return {candidate.id: float(weight) for candidate, weight in zip(candidates, weights)}

    def _ledger_payload(self, ledger: ContextLedger | None) -> dict[str, Any] | None:
        if ledger is None:
            return None
        return ledger.to_dict()

    def _sorted_factors(self, factors: set[str]) -> list[str]:
        return [factor for factor in FACTOR_ORDER if factor in factors]

    def _candidate_rows(
        self,
        *,
        effective_roles: list[str],
        open_results: list[Chunk],
        scoped_results: list[Chunk],
        candidates: list[Chunk],
        selected: list[Chunk],
        weights: list[float],
        selection_budget: int,
        prior_chunks: list[Chunk],
    ) -> list[dict[str, Any]]:
        open_rank = {chunk.id: index + 1 for index, chunk in enumerate(open_results)}
        scoped_rank = {chunk.id: index + 1 for index, chunk in enumerate(scoped_results)}
        candidate_map: dict[str, Chunk] = {}
        for chunk in [*open_results, *scoped_results, *candidates]:
            candidate_map.setdefault(chunk.id, chunk)

        selected_ids = {chunk.id for chunk in selected}
        selected_total = sum(chunk.token_count for chunk in selected)
        available_after_selection = max(selection_budget - selected_total, 0)
        weight_map = self._weights_by_id(candidates, weights)
        rows: list[dict[str, Any]] = []

        for chunk_id, chunk in candidate_map.items():
            comparison_selected = [item for item in selected if item.id != chunk_id]
            within_overlap = max(
                (cosine_similarity(chunk.embedding, item.embedding) for item in comparison_selected),
                default=0.0,
            )
            cross_overlap = max(
                (cosine_similarity(chunk.embedding, item.embedding) for item in prior_chunks),
                default=0.0,
            )
            permitted = roles_permit(chunk.roles, effective_roles)
            factors: set[str] = set()
            if not permitted:
                factors.add("not_permitted")
            elif chunk_id not in selected_ids:
                if chunk.token_count > available_after_selection:
                    factors.add("budget_limited")
                if within_overlap >= WITHIN_TURN_FACTOR_THRESHOLD:
                    factors.add("high_within_turn_overlap")
                if prior_chunks and cross_overlap >= CROSS_TURN_FACTOR_THRESHOLD:
                    factors.add("high_cross_turn_overlap")
                if weight_map.get(chunk_id, 0.0) > 0.0:
                    factors.add("not_selected_after_rounding")
                if not factors:
                    factors.add("lower_ranked")
            rows.append(
                {
                    "phase": "scoped" if chunk_id in scoped_rank else "open",
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "title": chunk.metadata.get("title", chunk.document_id),
                    "roles": list(chunk.roles),
                    "permitted": permitted,
                    "selected": chunk_id in selected_ids,
                    "open_rank": open_rank.get(chunk_id),
                    "scoped_rank": scoped_rank.get(chunk_id),
                    "bm25_score": chunk.metadata.get("bm25_score"),
                    "semantic_score": chunk.metadata.get("semantic_score"),
                    "hybrid_score": chunk.metadata.get("hybrid_score", chunk.relevance),
                    "token_count": chunk.token_count,
                    "within_turn_overlap": round(within_overlap, 6),
                    "cross_turn_overlap": round(cross_overlap, 6),
                    "factors": self._sorted_factors(factors),
                    "metadata": dict(chunk.metadata),
                }
            )
        rows.sort(key=lambda row: (row.get("scoped_rank") or 10**6, row.get("open_rank") or 10**6, row["chunk_id"]))
        return rows

    def _selected_chunk_payload(self, selected: list[Chunk], include_text: bool = False) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for chunk in selected:
            row = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "title": chunk.metadata.get("title", chunk.document_id),
                "roles": list(chunk.roles),
                "token_count": chunk.token_count,
                "metadata": dict(chunk.metadata),
            }
            if include_text:
                row["text"] = chunk.text
            payload.append(row)
        return payload

    def _build_trace(
        self,
        *,
        audit_id: str,
        principal: Principal,
        request: QueryRequest,
        effective_roles: list[str],
        status: str,
        signal: Any,
        answer: str,
        response: str,
        selected: list[Chunk],
        tokens_used: int,
        remaining_budget: int | None,
        timings_ms: dict[str, float],
        candidates: list[Chunk],
        weights: list[float],
        prior_chunks: list[Chunk],
        conversation_id: str | None,
        ledger: ContextLedger | None,
        turn: int,
        selection_budget: int,
    ) -> AuditTrace:
        candidate_rows = self._candidate_rows(
            effective_roles=effective_roles,
            open_results=signal.open_results,
            scoped_results=signal.scoped_results,
            candidates=candidates,
            selected=selected,
            weights=weights,
            selection_budget=selection_budget,
            prior_chunks=prior_chunks,
        )
        return AuditTrace(
            audit_id=audit_id,
            created_at=datetime.now(UTC).isoformat(),
            principal=principal.to_dict(),
            request={
                "query": request.query,
                "strategy": request.strategy,
                "candidate_k": request.candidate_k,
                "token_budget": request.token_budget,
                "available_budget": selection_budget,
                "requested_roles": list(request.requested_roles),
                "effective_roles": list(effective_roles),
                "conversation_id": conversation_id,
                "turn": turn,
                "lam": request.lam,
                "cross_turn_lambda": request.cross_turn_lambda,
                "embedding_model": self.embedding_model,
            },
            outcome={
                "status": status,
                "classification": signal.classification,
                "response": response,
                "answer": answer,
                "selected_chunk_ids": [chunk.id for chunk in selected],
                "selected_chunk_titles": [chunk.metadata.get("title", chunk.document_id) for chunk in selected],
                "hints": list(signal.hints),
                "required_role": signal.required_role,
                "tokens_used": tokens_used,
                "remaining_budget": remaining_budget,
            },
            classifier={
                "open_score": signal.open_score,
                "scoped_score": signal.scoped_score,
                "gap": signal.gap,
                "topic_support": signal.topic_support,
                "thresholds": self.thresholds.to_dict(),
            },
            timings=timings_ms,
            candidates=candidate_rows,
            conversation={
                "conversation_id": conversation_id,
                "total_budget": getattr(ledger, "total_budget", request.token_budget),
                "tokens_used": getattr(ledger, "tokens_used", 0),
                "remaining_budget": getattr(ledger, "remaining_budget", remaining_budget or 0),
                "ledger": self._ledger_payload(ledger),
            }
            if conversation_id and ledger is not None
            else None,
            selected_chunks=self._selected_chunk_payload(selected),
        )

    def _record_trace(self, trace: AuditTrace) -> None:
        if self.audit_store is not None:
            self.audit_store.record_trace(trace)

    def query(
        self,
        request: QueryRequest,
        *,
        principal: Principal | None = None,
        transport: str = "cli",
    ) -> QueryResponse:
        """Run the full MOSAIC runtime for one request and return a structured response."""
        if not request.query.strip():
            raise ValueError("Query must not be empty.")
        if request.candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")
        if request.token_budget <= 0:
            raise ValueError("token_budget must be positive.")

        audit_id = uuid4().hex
        started = perf_counter()

        resolved_principal, effective_roles = self._resolve_principal_and_roles(request, principal, transport)
        use_ledger = request.strategy == "mosaic_full"
        if use_ledger:
            conversation_id, ledger = self._load_ledger(request.conversation_id, request.token_budget)
        else:
            conversation_id, ledger = None, None
        prior_chunks = self._prior_chunks(ledger)
        selection_budget = self._selection_budget(request, ledger)
        turn = self._resolve_turn(request.turn, ledger)

        # Engine 2: decide whether the query is answerable, blocked by permissions, or a retrieval miss.
        signal_started = perf_counter()
        signal = classify_query(
            query=request.query,
            chunks=self.index.chunks,
            user_roles=effective_roles,
            thresholds=self.thresholds,
            candidate_k=request.candidate_k,
            embedding_model=self.embedding_model,
            corpus_idf=self.corpus_idf,
            measure_overhead=True,
        )
        signal_ms = (perf_counter() - signal_started) * 1000.0
        status = STATUS_BY_CLASSIFICATION.get(signal.classification, signal.classification.lower())

        if signal.classification != FAILURE_SUCCESS:
            remaining_budget = ledger.remaining_budget if ledger is not None else request.token_budget
            timings_ms = {
                "classification": round(signal_ms, 3),
                "retrieval": round(signal.latency_ms, 3),
                "overhead": round(signal.overhead_ms, 3),
                "optimization": 0.0,
                "answer": 0.0,
                "total": round((perf_counter() - started) * 1000.0, 3),
            }
            trace = self._build_trace(
                audit_id=audit_id,
                principal=resolved_principal,
                request=request,
                effective_roles=effective_roles,
                status=status,
                signal=signal,
                answer="",
                response=signal.response,
                selected=[],
                tokens_used=0,
                remaining_budget=remaining_budget,
                timings_ms=timings_ms,
                candidates=signal.scoped_results,
                weights=[],
                prior_chunks=prior_chunks,
                conversation_id=conversation_id,
                ledger=ledger,
                turn=turn,
                selection_budget=selection_budget,
            )
            self._record_trace(trace)
            return QueryResponse(
                status=status,
                answer="",
                response=signal.response,
                classification=signal.classification,
                selected_chunk_ids=[],
                selected_chunk_titles=[],
                hints=list(signal.hints),
                required_role=signal.required_role,
                audit_id=audit_id,
                conversation_id=conversation_id,
                tokens_used=0,
                remaining_budget=remaining_budget,
                timings_ms=timings_ms,
                trace=trace.to_dict() if request.return_trace else None,
            )

        # Engine 1: assemble minimal-overlap context under the active budget.
        candidates = signal.scoped_results or hybrid_retrieve(
            request.query,
            self.index.chunks,
            effective_roles,
            request.candidate_k,
            0.65,
            self.embedding_model,
        )
        optimization_started = perf_counter()
        result = select_strategy(
            request.strategy,
            candidates,
            user_roles=effective_roles,
            token_budget=selection_budget,
            lam=request.lam,
            use_jax=None,
            ledger=ledger if use_ledger else None,
            cross_turn_lambda=request.cross_turn_lambda,
        )
        optimization_ms = (perf_counter() - optimization_started) * 1000.0

        answer_started = perf_counter()
        context = build_context(prior_chunks + result.selected)
        answer = answer_query_locally(context, request.query)
        answer_ms = (perf_counter() - answer_started) * 1000.0

        if ledger is not None:
            ledger.add_chunks(result.selected, turn=turn, query_id=request.query)
            if conversation_id is not None and isinstance(self.ledger_store, JsonLedgerStore):
                self.ledger_store.save(
                    conversation_id,
                    ledger,
                    audit_id=audit_id,
                    principal_id=resolved_principal.principal_id,
                    turn=turn,
                )

        remaining_budget = ledger.remaining_budget if ledger is not None else result.remaining_budget
        timings_ms = {
            "classification": round(signal_ms, 3),
            "retrieval": round(signal.latency_ms, 3),
            "overhead": round(signal.overhead_ms, 3),
            "optimization": round(optimization_ms, 3),
            "answer": round(answer_ms, 3),
            "total": round((perf_counter() - started) * 1000.0, 3),
        }
        trace = self._build_trace(
            audit_id=audit_id,
            principal=resolved_principal,
            request=request,
            effective_roles=effective_roles,
            status=status,
            signal=signal,
            answer=answer,
            response=answer,
            selected=result.selected,
            tokens_used=result.tokens_used,
            remaining_budget=remaining_budget,
            timings_ms=timings_ms,
            candidates=candidates,
            weights=result.weights,
            prior_chunks=prior_chunks,
            conversation_id=conversation_id,
            ledger=ledger,
            turn=turn,
            selection_budget=selection_budget,
        )
        self._record_trace(trace)
        return QueryResponse(
            status=status,
            answer=answer,
            response=answer,
            classification=signal.classification,
            selected_chunk_ids=[chunk.id for chunk in result.selected],
            selected_chunk_titles=[chunk.metadata.get("title", chunk.document_id) for chunk in result.selected],
            hints=list(signal.hints),
            required_role=signal.required_role,
            audit_id=audit_id,
            conversation_id=conversation_id,
            tokens_used=result.tokens_used,
            remaining_budget=remaining_budget,
            timings_ms=timings_ms,
            trace=trace.to_dict() if request.return_trace else None,
        )

    def get_audit_trace(
        self,
        audit_id: str,
        *,
        include_text: bool = False,
        candidate_limit: int | None = None,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        if self.audit_store is None:
            raise RuntimeError("Audit store is not configured.")
        return self.audit_store.get_trace(
            audit_id,
            chunk_lookup=self.chunk_lookup,
            include_text=include_text,
            candidate_limit=candidate_limit,
            summary_only=summary_only,
        )

    def list_audit_events(self, **filters: Any) -> list[dict[str, Any]]:
        if self.audit_store is None:
            raise RuntimeError("Audit store is not configured.")
        return self.audit_store.list_events(**filters)

    def export_audit(
        self,
        *,
        audit_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
        include_text: bool = False,
    ) -> dict[str, Any]:
        if self.audit_store is None:
            raise RuntimeError("Audit store is not configured.")
        return self.audit_store.export(
            audit_id=audit_id,
            conversation_id=conversation_id,
            limit=limit,
            chunk_lookup=self.chunk_lookup,
            include_text=include_text,
        )
