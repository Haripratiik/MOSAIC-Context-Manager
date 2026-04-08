from __future__ import annotations

"""Audit persistence, export helpers, and SQLite-backed conversation state for MOSAIC."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import json
import sqlite3
from typing import Any

from .types import Chunk

FACTOR_ORDER = [
    "not_permitted",
    "budget_limited",
    "high_cross_turn_overlap",
    "high_within_turn_overlap",
    "not_selected_after_rounding",
    "lower_ranked",
]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def compact_ledger_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    embeddings = compact.pop("selected_embeddings", None)
    if isinstance(embeddings, list):
        compact["selected_embedding_count"] = len(embeddings)
    return compact


def _candidate_digest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id"),
        "title": row.get("title"),
        "selected": row.get("selected", False),
        "permitted": row.get("permitted", True),
        "scoped_rank": row.get("scoped_rank"),
        "open_rank": row.get("open_rank"),
        "hybrid_score": row.get("hybrid_score"),
        "token_count": row.get("token_count"),
        "factors": list(row.get("factors", [])),
    }


def _selected_chunk_digest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id"),
        "document_id": row.get("document_id"),
        "title": row.get("title"),
        "roles": list(row.get("roles", [])),
        "token_count": row.get("token_count"),
    }


def _trace_summary(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = list(payload.get("candidates", []))
    selected_chunks = list(payload.get("selected_chunks", []))
    denied_examples = list(payload.get("denied_candidate_examples", [_candidate_digest(row) for row in candidates if not row.get("permitted", True)][:5]))
    conversation = payload.get("conversation") or {}
    snapshots = conversation.get("snapshots", []) if isinstance(conversation, dict) else []
    latest_snapshot = snapshots[-1] if snapshots else None
    return {
        "audit_id": payload.get("audit_id"),
        "created_at": payload.get("created_at"),
        "principal": payload.get("principal"),
        "request": payload.get("request"),
        "outcome": payload.get("outcome"),
        "timings": payload.get("timings"),
        "candidate_count": payload.get("candidate_count", len(candidates)),
        "candidates_truncated": payload.get("candidates_truncated", False),
        "top_candidates": [_candidate_digest(row) for row in candidates[:5]],
        "denied_candidate_count": payload.get("denied_candidate_count", sum(1 for row in candidates if not row.get("permitted", True))),
        "denied_examples": denied_examples,
        "selected_chunks": [_selected_chunk_digest(row) for row in selected_chunks],
        "conversation": {
            "conversation_id": conversation.get("conversation_id"),
            "snapshot_count": len(snapshots),
            "latest_snapshot": latest_snapshot,
        }
        if conversation
        else None,
    }


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


@contextmanager
def _connection(path: str | Path):
    connection = _connect(path)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def ensure_schema(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _connection(target) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_requests (
                audit_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                transport TEXT NOT NULL,
                principal_id TEXT,
                principal_name TEXT,
                auth_source TEXT,
                requested_roles TEXT NOT NULL,
                effective_roles TEXT NOT NULL,
                query TEXT NOT NULL,
                strategy TEXT NOT NULL,
                candidate_k INTEGER NOT NULL,
                token_budget INTEGER NOT NULL,
                conversation_id TEXT,
                turn INTEGER,
                lam REAL NOT NULL,
                cross_turn_lambda REAL NOT NULL,
                embedding_model TEXT NOT NULL,
                status TEXT NOT NULL,
                classification TEXT NOT NULL,
                response TEXT,
                answer TEXT,
                selected_chunk_ids TEXT NOT NULL,
                selected_chunk_titles TEXT NOT NULL,
                hints TEXT NOT NULL,
                required_role TEXT,
                open_score REAL,
                scoped_score REAL,
                gap REAL,
                topic_support REAL,
                thresholds TEXT NOT NULL,
                timings TEXT NOT NULL,
                tokens_used INTEGER NOT NULL,
                remaining_budget INTEGER,
                denied_candidates INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS audit_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                chunk_roles TEXT NOT NULL,
                permitted INTEGER NOT NULL,
                selected INTEGER NOT NULL,
                open_rank INTEGER,
                scoped_rank INTEGER,
                bm25_score REAL,
                semantic_score REAL,
                hybrid_score REAL,
                token_count INTEGER NOT NULL,
                within_turn_overlap REAL,
                cross_turn_overlap REAL,
                factors TEXT NOT NULL,
                metadata TEXT NOT NULL,
                UNIQUE(audit_id, chunk_id),
                FOREIGN KEY(audit_id) REFERENCES audit_requests(audit_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS audit_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                audit_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                principal_id TEXT,
                turn INTEGER,
                total_budget INTEGER NOT NULL,
                tokens_used INTEGER NOT NULL,
                remaining_budget INTEGER NOT NULL,
                ledger_json TEXT NOT NULL,
                selected_chunk_ids TEXT NOT NULL,
                FOREIGN KEY(audit_id) REFERENCES audit_requests(audit_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_audit_requests_created_at ON audit_requests(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_requests_conversation_id ON audit_requests(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_audit_requests_status ON audit_requests(status);
            CREATE INDEX IF NOT EXISTS idx_audit_requests_classification ON audit_requests(classification);
            CREATE INDEX IF NOT EXISTS idx_audit_candidates_audit_id ON audit_candidates(audit_id);
            CREATE INDEX IF NOT EXISTS idx_audit_conversations_conversation_id ON audit_conversations(conversation_id, id DESC);
            """
        )


@dataclass(slots=True)
class AuditTrace:
    audit_id: str
    created_at: str
    principal: dict[str, Any]
    request: dict[str, Any]
    outcome: dict[str, Any]
    classifier: dict[str, Any]
    timings: dict[str, float]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    conversation: dict[str, Any] | None = None
    selected_chunks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        conversation = self.conversation
        if conversation and isinstance(conversation.get("ledger"), dict):
            conversation = dict(conversation)
            conversation["ledger"] = compact_ledger_payload(conversation["ledger"])
        return {
            "audit_id": self.audit_id,
            "created_at": self.created_at,
            "principal": self.principal,
            "request": self.request,
            "outcome": self.outcome,
            "classifier": self.classifier,
            "timings": self.timings,
            "candidates": self.candidates,
            "conversation": conversation,
            "selected_chunks": self.selected_chunks,
        }


class SQLiteAuditStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_schema(self.path)

    def record_trace(self, trace: AuditTrace) -> None:
        denied_candidates = sum(1 for row in trace.candidates if not row.get("permitted", True))
        with _connection(self.path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO audit_requests (
                    audit_id, created_at, transport, principal_id, principal_name, auth_source,
                    requested_roles, effective_roles, query, strategy, candidate_k, token_budget,
                    conversation_id, turn, lam, cross_turn_lambda, embedding_model, status,
                    classification, response, answer, selected_chunk_ids, selected_chunk_titles,
                    hints, required_role, open_score, scoped_score, gap, topic_support,
                    thresholds, timings, tokens_used, remaining_budget, denied_candidates
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.audit_id,
                    trace.created_at,
                    str(trace.principal.get("transport", "unknown")),
                    trace.principal.get("principal_id"),
                    trace.principal.get("display_name"),
                    trace.principal.get("auth_source"),
                    _json(trace.request.get("requested_roles", [])),
                    _json(trace.request.get("effective_roles", [])),
                    trace.request.get("query"),
                    trace.request.get("strategy"),
                    int(trace.request.get("candidate_k", 0)),
                    int(trace.request.get("token_budget", 0)),
                    trace.request.get("conversation_id"),
                    trace.request.get("turn"),
                    float(trace.request.get("lam", 0.0)),
                    float(trace.request.get("cross_turn_lambda", 0.0)),
                    trace.request.get("embedding_model", "unknown"),
                    trace.outcome.get("status", "unknown"),
                    trace.outcome.get("classification", "unknown"),
                    trace.outcome.get("response", ""),
                    trace.outcome.get("answer", ""),
                    _json(trace.outcome.get("selected_chunk_ids", [])),
                    _json(trace.outcome.get("selected_chunk_titles", [])),
                    _json(trace.outcome.get("hints", [])),
                    trace.outcome.get("required_role"),
                    float(trace.classifier.get("open_score", 0.0)),
                    float(trace.classifier.get("scoped_score", 0.0)),
                    float(trace.classifier.get("gap", 0.0)),
                    float(trace.classifier.get("topic_support", 0.0)),
                    _json(trace.classifier.get("thresholds", {})),
                    _json(trace.timings),
                    int(trace.outcome.get("tokens_used", 0)),
                    trace.outcome.get("remaining_budget"),
                    denied_candidates,
                ),
            )
            connection.execute("DELETE FROM audit_candidates WHERE audit_id = ?", (trace.audit_id,))
            connection.executemany(
                """
                INSERT INTO audit_candidates (
                    audit_id, phase, chunk_id, document_id, title, chunk_roles, permitted, selected,
                    open_rank, scoped_rank, bm25_score, semantic_score, hybrid_score, token_count,
                    within_turn_overlap, cross_turn_overlap, factors, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace.audit_id,
                        row.get("phase", "candidate"),
                        row.get("chunk_id"),
                        row.get("document_id"),
                        row.get("title", row.get("document_id", "unknown")),
                        _json(row.get("roles", [])),
                        1 if row.get("permitted", False) else 0,
                        1 if row.get("selected", False) else 0,
                        row.get("open_rank"),
                        row.get("scoped_rank"),
                        row.get("bm25_score"),
                        row.get("semantic_score"),
                        row.get("hybrid_score"),
                        int(row.get("token_count", 0)),
                        row.get("within_turn_overlap"),
                        row.get("cross_turn_overlap"),
                        _json(row.get("factors", [])),
                        _json(row.get("metadata", {})),
                    )
                    for row in trace.candidates
                ],
            )
            if trace.conversation and trace.request.get("conversation_id"):
                connection.execute(
                    """
                    INSERT INTO audit_conversations (
                        conversation_id, audit_id, created_at, principal_id, turn, total_budget,
                        tokens_used, remaining_budget, ledger_json, selected_chunk_ids
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.request.get("conversation_id"),
                        trace.audit_id,
                        trace.created_at,
                        trace.principal.get("principal_id"),
                        trace.request.get("turn"),
                        int(trace.conversation.get("total_budget", trace.request.get("token_budget", 0))),
                        int(trace.conversation.get("tokens_used", 0)),
                        int(trace.conversation.get("remaining_budget", 0)),
                        _json(trace.conversation.get("ledger", {})),
                        _json(trace.outcome.get("selected_chunk_ids", [])),
                    ),
                )

    def _row_to_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "audit_id": row["audit_id"],
            "created_at": row["created_at"],
            "principal_id": row["principal_id"],
            "transport": row["transport"],
            "query": row["query"],
            "strategy": row["strategy"],
            "status": row["status"],
            "classification": row["classification"],
            "conversation_id": row["conversation_id"],
            "turn": row["turn"],
            "tokens_used": row["tokens_used"],
            "remaining_budget": row["remaining_budget"],
            "required_role": row["required_role"],
            "denied_candidates": row["denied_candidates"],
        }

    def stats(self) -> dict[str, Any]:
        with _connection(self.path) as connection:
            request_count = int(connection.execute("SELECT COUNT(*) FROM audit_requests").fetchone()[0])
            conversation_count = int(connection.execute("SELECT COUNT(DISTINCT conversation_id) FROM audit_requests WHERE conversation_id IS NOT NULL").fetchone()[0])
            latest_created_at = connection.execute("SELECT created_at FROM audit_requests ORDER BY created_at DESC LIMIT 1").fetchone()
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM audit_requests GROUP BY status ORDER BY count DESC, status"
            ).fetchall()
        return {
            "request_count": request_count,
            "conversation_count": conversation_count,
            "latest_created_at": latest_created_at[0] if latest_created_at else None,
            "status_counts": [{"status": row["status"], "count": row["count"]} for row in status_rows],
        }

    def list_events(
        self,
        *,
        limit: int = 20,
        status: str | None = None,
        classification: str | None = None,
        principal_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT audit_id, created_at, principal_id, transport, query, strategy, status, classification, "
            "conversation_id, turn, tokens_used, remaining_budget, required_role, denied_candidates "
            "FROM audit_requests WHERE 1 = 1"
        )
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if classification:
            query += " AND classification = ?"
            params.append(classification)
        if principal_id:
            query += " AND principal_id = ?"
            params.append(principal_id)
        if conversation_id:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with _connection(self.path) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def get_trace(
        self,
        audit_id: str,
        *,
        chunk_lookup: dict[str, Chunk] | None = None,
        include_text: bool = False,
        candidate_limit: int | None = None,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        with _connection(self.path) as connection:
            request_row = connection.execute("SELECT * FROM audit_requests WHERE audit_id = ?", (audit_id,)).fetchone()
            if request_row is None:
                raise KeyError(f"Unknown audit id: {audit_id}")
            candidate_rows = connection.execute(
                "SELECT * FROM audit_candidates WHERE audit_id = ? ORDER BY COALESCE(scoped_rank, 1000000), COALESCE(open_rank, 1000000), id",
                (audit_id,),
            ).fetchall()
            conversation_rows = []
            if request_row["conversation_id"]:
                conversation_rows = connection.execute(
                    "SELECT * FROM audit_conversations WHERE conversation_id = ? ORDER BY id",
                    (request_row["conversation_id"],),
                ).fetchall()

        selected_ids = _loads(request_row["selected_chunk_ids"], [])
        selected_chunks: list[dict[str, Any]] = []
        candidate_payload: list[dict[str, Any]] = []
        metadata_by_id: dict[str, dict[str, Any]] = {}
        for row in candidate_rows:
            payload = {
                "phase": row["phase"],
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "roles": _loads(row["chunk_roles"], []),
                "permitted": bool(row["permitted"]),
                "selected": bool(row["selected"]),
                "open_rank": row["open_rank"],
                "scoped_rank": row["scoped_rank"],
                "bm25_score": row["bm25_score"],
                "semantic_score": row["semantic_score"],
                "hybrid_score": row["hybrid_score"],
                "token_count": row["token_count"],
                "within_turn_overlap": row["within_turn_overlap"],
                "cross_turn_overlap": row["cross_turn_overlap"],
                "factors": _loads(row["factors"], []),
                "metadata": _loads(row["metadata"], {}),
            }
            metadata_by_id[payload["chunk_id"]] = payload
            candidate_payload.append(payload)
        for chunk_id in selected_ids:
            payload = dict(metadata_by_id.get(chunk_id, {"chunk_id": chunk_id, "metadata": {}}))
            if chunk_lookup and chunk_id in chunk_lookup:
                payload["metadata"] = dict(chunk_lookup[chunk_id].metadata)
                payload["document_id"] = chunk_lookup[chunk_id].document_id
                payload["title"] = chunk_lookup[chunk_id].metadata.get("title", chunk_lookup[chunk_id].document_id)
                payload["roles"] = list(chunk_lookup[chunk_id].roles)
                payload["token_count"] = chunk_lookup[chunk_id].token_count
                if include_text:
                    payload["text"] = chunk_lookup[chunk_id].text
            selected_chunks.append(payload)

        trace = AuditTrace(
            audit_id=request_row["audit_id"],
            created_at=request_row["created_at"],
            principal={
                "principal_id": request_row["principal_id"],
                "display_name": request_row["principal_name"],
                "transport": request_row["transport"],
                "auth_source": request_row["auth_source"],
            },
            request={
                "query": request_row["query"],
                "strategy": request_row["strategy"],
                "candidate_k": request_row["candidate_k"],
                "token_budget": request_row["token_budget"],
                "requested_roles": _loads(request_row["requested_roles"], []),
                "effective_roles": _loads(request_row["effective_roles"], []),
                "conversation_id": request_row["conversation_id"],
                "turn": request_row["turn"],
                "lam": request_row["lam"],
                "cross_turn_lambda": request_row["cross_turn_lambda"],
                "embedding_model": request_row["embedding_model"],
            },
            outcome={
                "status": request_row["status"],
                "classification": request_row["classification"],
                "response": request_row["response"],
                "answer": request_row["answer"],
                "selected_chunk_ids": selected_ids,
                "selected_chunk_titles": _loads(request_row["selected_chunk_titles"], []),
                "hints": _loads(request_row["hints"], []),
                "required_role": request_row["required_role"],
                "tokens_used": request_row["tokens_used"],
                "remaining_budget": request_row["remaining_budget"],
            },
            classifier={
                "open_score": request_row["open_score"],
                "scoped_score": request_row["scoped_score"],
                "gap": request_row["gap"],
                "topic_support": request_row["topic_support"],
                "thresholds": _loads(request_row["thresholds"], {}),
            },
            timings=_loads(request_row["timings"], {}),
            candidates=candidate_payload,
            conversation={
                "conversation_id": request_row["conversation_id"],
                "snapshots": [
                    {
                        "audit_id": row["audit_id"],
                        "turn": row["turn"],
                        "tokens_used": row["tokens_used"],
                        "remaining_budget": row["remaining_budget"],
                        "ledger": compact_ledger_payload(_loads(row["ledger_json"], {})),
                        "selected_chunk_ids": _loads(row["selected_chunk_ids"], []),
                    }
                    for row in conversation_rows
                ],
            }
            if request_row["conversation_id"]
            else None,
            selected_chunks=selected_chunks,
        )
        payload = trace.to_dict()
        total_candidates = len(payload.get("candidates", []))
        payload["candidate_count"] = total_candidates
        payload["denied_candidate_count"] = sum(1 for row in payload.get("candidates", []) if not row.get("permitted", True))
        denied_candidate_examples = [
            _candidate_digest(row)
            for row in payload.get("candidates", [])
            if not row.get("permitted", True)
        ][:5]
        payload["candidates_truncated"] = False
        if candidate_limit is not None and candidate_limit >= 0 and total_candidates > candidate_limit:
            payload["candidates"] = payload["candidates"][:candidate_limit]
            payload["candidates_truncated"] = True
        if summary_only:
            payload["denied_candidate_examples"] = denied_candidate_examples
            return _trace_summary(payload)
        return payload

    def export(
        self,
        *,
        audit_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
        chunk_lookup: dict[str, Chunk] | None = None,
        include_text: bool = False,
    ) -> dict[str, Any]:
        if audit_id:
            return {"mode": "request", "request": self.get_trace(audit_id, chunk_lookup=chunk_lookup, include_text=include_text)}
        if conversation_id:
            request_summaries = self.list_events(limit=1000, conversation_id=conversation_id)
            return {
                "mode": "conversation",
                "conversation_id": conversation_id,
                "requests": [
                    self.get_trace(summary["audit_id"], chunk_lookup=chunk_lookup, include_text=include_text)
                    for summary in sorted(request_summaries, key=lambda item: (item.get("turn") or 0, item["created_at"]))
                ],
            }
        return {
            "mode": "batch",
            "requests": [
                self.get_trace(summary["audit_id"], chunk_lookup=chunk_lookup, include_text=include_text)
                for summary in self.list_events(limit=limit)
            ],
        }


class SQLiteLedgerStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_schema(self.path)

    def load(self, conversation_id: str, total_budget: int, factory: callable) -> Any:
        with _connection(self.path) as connection:
            row = connection.execute(
                "SELECT ledger_json FROM audit_conversations WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return factory(total_budget=total_budget)
        return factory.from_dict(_loads(row["ledger_json"], {"total_budget": total_budget}))

    def save(self, conversation_id: str, ledger: Any, *, audit_id: str | None = None, principal_id: str | None = None, turn: int | None = None) -> None:
        with _connection(self.path) as connection:
            connection.execute(
                """
                INSERT INTO audit_conversations (
                    conversation_id, audit_id, created_at, principal_id, turn, total_budget,
                    tokens_used, remaining_budget, ledger_json, selected_chunk_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    audit_id or "pending",
                    _utcnow(),
                    principal_id,
                    turn,
                    int(getattr(ledger, "total_budget", 0)),
                    int(getattr(ledger, "tokens_used", 0)),
                    int(getattr(ledger, "remaining_budget", 0)),
                    _json(ledger.to_dict()),
                    _json(list(getattr(ledger, "selected_chunk_ids", []))),
                ),
            )


class JsonLedgerStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, conversation_id: str, total_budget: int, factory: callable) -> Any:
        if not self.path.exists():
            return factory(total_budget=total_budget)
        return factory.from_dict(_loads(self.path.read_text(encoding="utf-8-sig"), {"total_budget": total_budget}))

    def save(self, conversation_id: str, ledger: Any, *, audit_id: str | None = None, principal_id: str | None = None, turn: int | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(_json(ledger.to_dict()), encoding="utf-8")


def build_governance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    requests = payload.get("requests", [])
    if payload.get("mode") == "request" and payload.get("request"):
        requests = [payload["request"]]
    outcome_counts: dict[str, int] = {}
    deny_matrix: dict[tuple[str, str], int] = {}
    denied_candidate_counts: list[int] = []
    sample_success = None
    sample_gap = None
    for trace in requests:
        status = str(trace.get("outcome", {}).get("status", "unknown"))
        outcome_counts[status] = outcome_counts.get(status, 0) + 1
        denied = sum(1 for row in trace.get("candidates", []) if not row.get("permitted", True))
        denied_candidate_counts.append(denied)
        required_role = trace.get("outcome", {}).get("required_role")
        caller = ",".join(trace.get("request", {}).get("effective_roles", [])) or "unknown"
        if status == "permission_gap" and required_role:
            deny_matrix[(caller, str(required_role))] = deny_matrix.get((caller, str(required_role)), 0) + 1
            if sample_gap is None:
                sample_gap = trace
        if status == "success" and sample_success is None:
            sample_success = trace
    return {
        "request_count": len(requests),
        "outcomes": [{"status": key, "count": value} for key, value in sorted(outcome_counts.items())],
        "deny_matrix": [
            {"caller_roles": caller, "required_role": role, "count": count}
            for (caller, role), count in sorted(deny_matrix.items())
        ],
        "avg_denied_candidates": round(sum(denied_candidate_counts) / max(len(denied_candidate_counts), 1), 3),
        "sample_success": sample_success,
        "sample_permission_gap": sample_gap,
    }


def load_governance_source(
    *,
    audit_db_path: str | Path | None = None,
    audit_export_path: str | Path | None = None,
    chunk_lookup: dict[str, Chunk] | None = None,
) -> dict[str, Any] | None:
    if audit_export_path:
        payload = json.loads(Path(audit_export_path).read_text(encoding="utf-8-sig"))
        return build_governance_summary(payload)
    if audit_db_path:
        store = SQLiteAuditStore(audit_db_path)
        payload = store.export(limit=100, chunk_lookup=chunk_lookup, include_text=False)
        return build_governance_summary(payload)
    return None
