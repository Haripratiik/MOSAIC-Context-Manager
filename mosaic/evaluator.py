from __future__ import annotations

"""Evaluation, provenance tracking, and benchmark orchestration for MOSAIC."""

import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ingestor import load_index
from .ledger import ContextLedger
from .optimizer import OptimizationResult, compute_redundancy_score, optimize, select_mmr, select_topk
from .retriever import hybrid_retrieve
from .signal import SignalThresholds, calibrate_thresholds, run_signal_eval
from .types import Chunk
from .utils import DEFAULT_EMBEDDING_MODEL, HASH_EMBEDDING_MODEL, build_context, clamp, cosine_similarity, count_tokens, dump_json, embed_texts, load_json, mean, sentencize, tokenize

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

DEFAULT_ANTHROPIC_ANSWER_MODEL = os.getenv("MOSAIC_ANSWER_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_ANTHROPIC_JUDGE_MODEL = os.getenv("MOSAIC_JUDGE_MODEL", "claude-haiku-4-5-20251001")

FAITHFULNESS_PROMPT = """
Given this context:
{context}

And this answer to the question "{query}":
{answer}

Rate from 0.0 to 1.0: is the answer fully supported by the context?
Respond with only a number.
""".strip()


def _suite_queries(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "queries" in payload:
        return list(payload["queries"])
    return list(payload)  # type: ignore[arg-type]


def compute_similarity_matrix(chunks: list[Chunk]) -> list[list[float]]:
    return [[round(cosine_similarity(left.embedding, right.embedding), 6) for right in chunks] for left in chunks] if chunks else []


def compute_query_coverage(query: str, chunks: list[Chunk], embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> float:
    if not chunks:
        return 0.0
    query_embedding = embed_texts([query], model_name=embedding_model, dimensions=len(chunks[0].embedding))[0]
    return sum(max(0.0, cosine_similarity(query_embedding, chunk.embedding)) for chunk in chunks)


def answer_query_locally(context: str, query: str, embedding_model: str = HASH_EMBEDDING_MODEL, max_sentences: int = 3) -> str:
    sentences = sentencize(context)
    if not sentences:
        return ""
    query_tokens = set(tokenize(query))
    embeddings = embed_texts([query, *sentences], model_name=embedding_model)
    query_embedding = embeddings[0]
    sentence_embeddings = embeddings[1:]
    ranked: list[tuple[float, str]] = []
    for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
        overlap = sum(1 for token in tokenize(sentence) if token in query_tokens)
        semantic = max(0.0, cosine_similarity(query_embedding, sentence_embedding))
        ranked.append((overlap + semantic, sentence))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return " ".join(sentence for _, sentence in ranked[:max_sentences]).strip()


def _proxy_faithfulness(context: str, answer: str, ground_truth_answer: str | None, embedding_model: str = HASH_EMBEDDING_MODEL) -> float:
    if not context:
        return 0.0
    context_embedding = embed_texts([context], model_name=embedding_model)[0]
    answer_embedding = embed_texts([answer or context[:200]], model_name=embedding_model)[0]
    support = max(0.0, cosine_similarity(context_embedding, answer_embedding))
    if ground_truth_answer:
        truth_embedding = embed_texts([ground_truth_answer], model_name=embedding_model)[0]
        truth_support = max(0.0, cosine_similarity(context_embedding, truth_embedding))
        answer_alignment = max(0.0, cosine_similarity(answer_embedding, truth_embedding))
        return clamp((0.45 * truth_support) + (0.35 * answer_alignment) + (0.20 * support))
    return clamp(support)


def _live_backend_ready(model_name: str | None) -> tuple[bool, str | None]:
    resolved_model = (model_name or '').strip()
    if not resolved_model:
        return False, 'model_not_configured'
    if anthropic is None:
        return False, 'anthropic_sdk_unavailable'
    if not os.getenv('ANTHROPIC_API_KEY'):
        return False, 'missing_api_key'
    return True, None


def backend_capabilities(answer_model: str | None = None, judge_model: str | None = None) -> dict[str, Any]:
    resolved_answer_model = answer_model or DEFAULT_ANTHROPIC_ANSWER_MODEL
    resolved_judge_model = judge_model or DEFAULT_ANTHROPIC_JUDGE_MODEL
    answer_ready, answer_reason = _live_backend_ready(resolved_answer_model)
    judge_ready, judge_reason = _live_backend_ready(resolved_judge_model)
    return {
        'anthropic_sdk_available': anthropic is not None,
        'anthropic_api_key_present': bool(os.getenv('ANTHROPIC_API_KEY')),
        'answer_model': resolved_answer_model,
        'judge_model': resolved_judge_model,
        'live_answer_ready': answer_ready,
        'live_answer_unavailable_reason': answer_reason,
        'live_judge_ready': judge_ready,
        'live_judge_unavailable_reason': judge_reason,
    }


def _raise_live_backend_error(label: str, model_name: str | None, reason: str | None) -> None:
    resolved_model = (model_name or '').strip() or '<unset>'
    detail = reason or 'unknown_reason'
    raise RuntimeError(f"Live {label} requested for model '{resolved_model}', but MOSAIC could not use it ({detail}).")


def measure_faithfulness(
    context: str,
    query: str,
    answer: str,
    ground_truth_answer: str | None = None,
    judge_model: str | None = None,
    require_live: bool = False,
) -> tuple[float, str, str | None]:
    resolved_model = judge_model or DEFAULT_ANTHROPIC_JUDGE_MODEL
    ready, reason = _live_backend_ready(resolved_model)
    fallback_reason = reason
    if ready:
        try:
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            response = client.messages.create(
                model=resolved_model,
                max_tokens=16,
                messages=[{'role': 'user', 'content': FAITHFULNESS_PROMPT.format(context=context, query=query, answer=answer)}],
            )
            text = response.content[0].text.strip()
            match = re.search(r'([01](?:\.\d+)?)', text)
            if match:
                return clamp(float(match.group(1))), 'anthropic', None
            fallback_reason = 'invalid_live_response'
        except Exception as exc:
            fallback_reason = f'live_request_failed:{exc.__class__.__name__}'
    if require_live:
        _raise_live_backend_error('judge backend', resolved_model, fallback_reason)
    return _proxy_faithfulness(context, answer, ground_truth_answer, HASH_EMBEDDING_MODEL), 'proxy', fallback_reason


def measure_ttft(
    context: str,
    query: str,
    answer_model: str | None = None,
    require_live: bool = False,
) -> tuple[float, str, str, str | None]:
    resolved_model = answer_model or DEFAULT_ANTHROPIC_ANSWER_MODEL
    ready, reason = _live_backend_ready(resolved_model)
    fallback_reason = reason
    if ready:
        prompt = f"Context:\n{context}\n\nAnswer concisely: {query}"
        try:
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            t0 = time.perf_counter()
            answer_parts: list[str] = []
            with client.messages.stream(model=resolved_model, max_tokens=256, messages=[{'role': 'user', 'content': prompt}]) as stream:
                first_token = None
                for index, text in enumerate(stream.text_stream):
                    if index == 0:
                        first_token = (time.perf_counter() - t0) * 1000.0
                    answer_parts.append(text)
            return round(first_token or 0.0, 3), ''.join(answer_parts).strip(), 'anthropic', None
        except Exception as exc:
            fallback_reason = f'live_request_failed:{exc.__class__.__name__}'
    if require_live:
        _raise_live_backend_error('answer backend', resolved_model, fallback_reason)
    answer = answer_query_locally(context, query, embedding_model=HASH_EMBEDDING_MODEL)
    proxy_ms = round(35.0 + (0.85 * count_tokens(context)) + (0.10 * len(tokenize(query))), 3)
    return proxy_ms, answer, 'proxy', fallback_reason


def summarize_eval_provenance(
    rows: list[dict[str, Any]],
    *,
    answer_model: str | None = None,
    judge_model: str | None = None,
    require_live_answer: bool = False,
    require_live_judge: bool = False,
) -> dict[str, Any]:
    capabilities = backend_capabilities(answer_model=answer_model, judge_model=judge_model)
    ttft_backend_counts: dict[str, int] = defaultdict(int)
    faithfulness_backend_counts: dict[str, int] = defaultdict(int)
    ttft_proxy_reason_counts: dict[str, int] = defaultdict(int)
    faithfulness_proxy_reason_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        ttft_backend_counts[str(row.get('ttft_backend', 'unknown'))] += 1
        faithfulness_backend_counts[str(row.get('fai_backend', 'unknown'))] += 1
        ttft_reason = row.get('ttft_backend_reason')
        fai_reason = row.get('fai_backend_reason')
        if ttft_reason:
            ttft_proxy_reason_counts[str(ttft_reason)] += 1
        if fai_reason:
            faithfulness_proxy_reason_counts[str(fai_reason)] += 1
    proxy_ttft = ttft_backend_counts.get('proxy', 0)
    proxy_fai = faithfulness_backend_counts.get('proxy', 0)
    return {
        **capabilities,
        'require_live_answer': require_live_answer,
        'require_live_judge': require_live_judge,
        'row_count': len(rows),
        'ttft_backend_counts': [
            {'backend': backend, 'count': count}
            for backend, count in sorted(ttft_backend_counts.items())
        ],
        'faithfulness_backend_counts': [
            {'backend': backend, 'count': count}
            for backend, count in sorted(faithfulness_backend_counts.items())
        ],
        'ttft_proxy_reason_counts': [
            {'reason': reason, 'count': count}
            for reason, count in sorted(ttft_proxy_reason_counts.items())
        ],
        'faithfulness_proxy_reason_counts': [
            {'reason': reason, 'count': count}
            for reason, count in sorted(faithfulness_proxy_reason_counts.items())
        ],
        'proxy_ttft_rows': proxy_ttft,
        'proxy_faithfulness_rows': proxy_fai,
        'all_live': proxy_ttft == 0 and proxy_fai == 0 and bool(rows),
    }


def _normalize_strategy(strategy: str) -> str:
    return "mosaic_no_ledger" if strategy == "mosaic" else strategy


def select_strategy(
    strategy: str,
    candidates: list[Chunk],
    user_roles: list[str],
    token_budget: int,
    lam: float,
    use_jax: bool | None = None,
    ledger: ContextLedger | None = None,
    cross_turn_lambda: float = 0.0,
) -> OptimizationResult:
    resolved = _normalize_strategy(strategy)
    if resolved in {"mosaic_no_ledger", "mosaic_full"}:
        return optimize(
            candidates,
            user_roles=user_roles,
            token_budget=token_budget,
            lam=lam,
            use_jax=use_jax,
            ledger=ledger if resolved == "mosaic_full" else None,
            cross_turn_lambda=cross_turn_lambda if resolved == "mosaic_full" else 0.0,
        )
    if resolved == "topk":
        selected = select_topk(candidates, token_budget=token_budget, user_roles=user_roles)
        return OptimizationResult(selected=selected, weights=[], tokens_used=sum(chunk.token_count for chunk in selected), objective_history=[], backend="topk", remaining_budget=max(0, token_budget - sum(chunk.token_count for chunk in selected)))
    if resolved == "mmr":
        selected = select_mmr(candidates, token_budget=token_budget, user_roles=user_roles)
        return OptimizationResult(selected=selected, weights=[], tokens_used=sum(chunk.token_count for chunk in selected), objective_history=[], backend="mmr", remaining_budget=max(0, token_budget - sum(chunk.token_count for chunk in selected)))
    raise ValueError(f"Unknown strategy: {strategy}")


def _prepare_candidates(index_chunks: list[Chunk], queries: list[dict[str, Any]], candidate_k: int, embedding_model: str) -> dict[str, tuple[list[Chunk], float]]:
    prepared: dict[str, tuple[list[Chunk], float]] = {}
    for item in queries:
        start = time.perf_counter()
        candidates = hybrid_retrieve(
            query=item["query"],
            chunks=index_chunks,
            user_roles=list(item.get("user_roles", ["public"])),
            k=candidate_k,
            alpha=0.65,
            embedding_model=embedding_model,
        )
        prepared[item["id"]] = (candidates, (time.perf_counter() - start) * 1000.0)
    return prepared


def _cross_turn_redundancy(selected: list[Chunk], prior_chunks: list[Chunk]) -> float:
    if not selected or not prior_chunks:
        return 0.0
    scores = []
    for chunk in selected:
        scores.append(max(cosine_similarity(chunk.embedding, prior.embedding) for prior in prior_chunks))
    return mean(scores)


def _evaluate_query(
    item: dict[str, Any],
    strategy: str,
    candidates: list[Chunk],
    retrieval_ms: float,
    token_budget: int,
    lam: float,
    embedding_model: str,
    answer_model: str | None,
    judge_model: str | None,
    use_jax: bool | None,
    ledger: ContextLedger | None = None,
    cross_turn_lambda: float = 0.0,
    prior_chunks: list[Chunk] | None = None,
    require_live_answer: bool = False,
    require_live_judge: bool = False,
) -> dict[str, Any]:
    query = item["query"]
    user_roles = list(item.get("user_roles", ["public"]))
    expected_docs = set(item.get("source_doc_ids", item.get("required_docs", [])))
    forbidden_docs = set(item.get("forbidden_doc_ids", []))

    optimize_start = time.perf_counter()
    result = select_strategy(
        strategy,
        candidates,
        user_roles=user_roles,
        token_budget=token_budget,
        lam=lam,
        use_jax=use_jax,
        ledger=ledger,
        cross_turn_lambda=cross_turn_lambda,
    )
    optimization_ms = (time.perf_counter() - optimize_start) * 1000.0
    previous_chunks = list(prior_chunks or [])
    context_chunks = previous_chunks + result.selected
    context = build_context(context_chunks)

    ttft_ms, answer, ttft_backend, ttft_backend_reason = measure_ttft(
        context,
        query,
        answer_model=answer_model,
        require_live=require_live_answer,
    )
    raw_faithfulness, fai_backend, fai_backend_reason = measure_faithfulness(
        context,
        query,
        answer,
        ground_truth_answer=item.get("ground_truth_answer"),
        judge_model=judge_model,
        require_live=require_live_judge,
    )

    selected_doc_ids = {chunk.document_id for chunk in result.selected}
    selected_chunk_ids = [chunk.id for chunk in result.selected]
    source_recall = len(selected_doc_ids & expected_docs) / len(expected_docs) if expected_docs else 0.0
    permission_violations = len(selected_doc_ids & forbidden_docs)
    coverage = compute_query_coverage(query, result.selected, embedding_model=embedding_model)
    redundancy = compute_redundancy_score(result.selected)
    cross_turn = _cross_turn_redundancy(result.selected, previous_chunks)
    tokens_used = sum(chunk.token_count for chunk in result.selected)
    faithfulness = clamp((0.65 * raw_faithfulness) + (0.35 * source_recall) - (0.25 * permission_violations))
    efficiency = faithfulness / max(tokens_used, 1)

    return {
        "query_id": item["id"],
        "category": item.get("category", "unknown"),
        "scenario_id": item.get("scenario_id"),
        "turn": item.get("turn"),
        "strategy": _normalize_strategy(strategy),
        "lam": lam,
        "query": query,
        "answer": answer,
        "ground_truth_answer": item.get("ground_truth_answer", ""),
        "user_roles": user_roles,
        "selected_doc_ids": sorted(selected_doc_ids),
        "selected_chunk_ids": selected_chunk_ids,
        "selected_chunk_titles": [chunk.metadata.get("title", chunk.document_id) for chunk in result.selected],
        "source_recall": round(source_recall, 4),
        "coverage_score": round(coverage, 4),
        "redundancy_score": round(redundancy, 4),
        "cross_turn_redundancy": round(cross_turn, 4),
        "faithfulness": round(faithfulness, 4),
        "ttft_ms": round(ttft_ms, 3),
        "tokens_used": tokens_used,
        "context_tokens_used": sum(chunk.token_count for chunk in context_chunks),
        "efficiency": round(efficiency, 6),
        "permission_violations": permission_violations,
        "optimizer_backend": result.backend,
        "fai_backend": fai_backend,
        "fai_backend_reason": fai_backend_reason,
        "ttft_backend": ttft_backend,
        "ttft_backend_reason": ttft_backend_reason,
        "answer_model": answer_model or DEFAULT_ANTHROPIC_ANSWER_MODEL,
        "judge_model": judge_model or DEFAULT_ANTHROPIC_JUDGE_MODEL,
        "similarity_matrix": compute_similarity_matrix(result.selected),
        "retrieval_ms": round(retrieval_ms, 3),
        "optimization_ms": round(optimization_ms, 3),
    }


def run_eval(
    index_path: str | Path,
    eval_suite_path: str | Path,
    strategy: str,
    token_budget: int = 4096,
    candidate_k: int = 50,
    lam: float = 0.5,
    answer_model: str | None = None,
    judge_model: str | None = None,
    use_jax: bool | None = None,
    output_path: str | Path | None = None,
    require_live_answer: bool = False,
    require_live_judge: bool = False,
) -> list[dict[str, Any]]:
    index = load_index(index_path)
    embedding_model = str(index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    eval_suite = load_json(eval_suite_path)
    queries = [item for item in _suite_queries(eval_suite) if item.get("category") not in {"failure_classification", "multi_turn"}]
    prepared = _prepare_candidates(index.chunks, queries, candidate_k, embedding_model)

    rows = [
        _evaluate_query(
            item=item,
            strategy=strategy,
            candidates=prepared[item["id"]][0],
            retrieval_ms=prepared[item["id"]][1],
            token_budget=token_budget,
            lam=lam,
            embedding_model=embedding_model,
            answer_model=answer_model,
            judge_model=judge_model,
            use_jax=use_jax,
            require_live_answer=require_live_answer,
            require_live_judge=require_live_judge,
        )
        for item in queries
    ]
    if output_path is not None:
        dump_json(output_path, rows)
    return rows


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy[row["strategy"]].append(row)

    strategy_rows = []
    category_rows = []
    for strategy, values in sorted(by_strategy.items()):
        strategy_rows.append(
            {
                "strategy": strategy,
                "faithfulness": round(mean([row["faithfulness"] for row in values]), 4),
                "coverage_score": round(mean([row["coverage_score"] for row in values]), 4),
                "redundancy_score": round(mean([row["redundancy_score"] for row in values]), 4),
                "ttft_ms": round(mean([row["ttft_ms"] for row in values]), 3),
                "tokens_used": round(mean([row["tokens_used"] for row in values]), 2),
                "efficiency": round(mean([row["efficiency"] for row in values]), 6),
                "source_recall": round(mean([row["source_recall"] for row in values]), 4),
                "permission_violations": round(mean([row["permission_violations"] for row in values]), 4),
                "retrieval_ms": round(mean([row.get("retrieval_ms", 0.0) for row in values]), 3),
                "optimization_ms": round(mean([row.get("optimization_ms", 0.0) for row in values]), 3),
            }
        )
        categories = sorted({row["category"] for row in values})
        for category in categories:
            category_values = [row for row in values if row["category"] == category]
            category_rows.append(
                {
                    "strategy": strategy,
                    "category": category,
                    "faithfulness": round(mean([row["faithfulness"] for row in category_values]), 4),
                    "coverage_score": round(mean([row["coverage_score"] for row in category_values]), 4),
                    "redundancy_score": round(mean([row["redundancy_score"] for row in category_values]), 4),
                    "ttft_ms": round(mean([row["ttft_ms"] for row in category_values]), 3),
                }
            )
    return {
        "strategies": strategy_rows,
        "categories": category_rows,
        "query_count": len(rows),
        "backend_provenance": summarize_eval_provenance(rows),
    }


def run_pareto(
    index_path: str | Path,
    eval_suite_path: str | Path,
    token_budget: int = 4096,
    candidate_k: int = 50,
    lam_values: list[float] | None = None,
    answer_model: str | None = None,
    judge_model: str | None = None,
    use_jax: bool | None = None,
    output_path: str | Path | None = None,
    require_live_answer: bool = False,
    require_live_judge: bool = False,
) -> list[dict[str, Any]]:
    lam_values = lam_values or [round(step * 0.1, 2) for step in range(0, 21)]
    index = load_index(index_path)
    embedding_model = str(index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    eval_suite = load_json(eval_suite_path)
    queries = [item for item in _suite_queries(eval_suite) if item.get("category") in {"redundancy_trap", "multi_hop"}]
    prepared = _prepare_candidates(index.chunks, queries, candidate_k, embedding_model)

    frontier: list[dict[str, Any]] = []
    for lam in lam_values:
        rows = [
            _evaluate_query(
                item=item,
                strategy="mosaic_no_ledger",
                candidates=prepared[item["id"]][0],
                retrieval_ms=prepared[item["id"]][1],
                token_budget=token_budget,
                lam=lam,
                embedding_model=embedding_model,
                answer_model=answer_model,
                judge_model=judge_model,
                use_jax=use_jax,
                require_live_answer=require_live_answer,
                require_live_judge=require_live_judge,
            )
            for item in queries
        ]
        summary = aggregate_results(rows)["strategies"][0]
        frontier.append(
            {
                "lam": lam,
                "faithfulness": summary["faithfulness"],
                "coverage_score": summary["coverage_score"],
                "redundancy_score": summary["redundancy_score"],
                "ttft_ms": summary["ttft_ms"],
                "tokens_used": summary["tokens_used"],
                "efficiency": summary["efficiency"],
            }
        )
    if output_path is not None:
        dump_json(output_path, frontier)
    return frontier


def _multiturn_turn_budget(conversation_budget: int, remaining_budget: int, turn_budget_cap: int | None = None) -> int:
    cap = turn_budget_cap if turn_budget_cap is not None else max(384, conversation_budget // 4)
    return max(0, min(remaining_budget, cap))


def run_multiturn_eval(
    index_path: str | Path,
    eval_suite_path: str | Path,
    strategy: str,
    conversation_budget: int = 2200,
    candidate_k: int = 50,
    turn_budget_cap: int | None = None,
    lam: float = 0.5,
    cross_turn_lambda: float = 0.75,
    answer_model: str | None = None,
    judge_model: str | None = None,
    use_jax: bool | None = None,
    output_path: str | Path | None = None,
    require_live_answer: bool = False,
    require_live_judge: bool = False,
) -> list[dict[str, Any]]:
    index = load_index(index_path)
    embedding_model = str(index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    eval_suite = load_json(eval_suite_path)
    queries = [item for item in _suite_queries(eval_suite) if item.get("category") == "multi_turn"]
    prepared = _prepare_candidates(index.chunks, queries, candidate_k, embedding_model)
    chunk_lookup = {chunk.id: chunk for chunk in index.chunks}

    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in queries:
        scenarios[item["scenario_id"]].append(item)
    for scenario_items in scenarios.values():
        scenario_items.sort(key=lambda item: int(item.get("turn", 0)))

    rows: list[dict[str, Any]] = []
    for scenario_id, scenario_items in sorted(scenarios.items()):
        ledger = ContextLedger(total_budget=conversation_budget)
        for item in scenario_items:
            prior_chunks = [chunk_lookup[chunk_id] for chunk_id in ledger.selected_chunk_ids if chunk_id in chunk_lookup]
            turn_budget = _multiturn_turn_budget(conversation_budget, ledger.remaining_budget, turn_budget_cap)
            row = _evaluate_query(
                item=item,
                strategy=strategy,
                candidates=prepared[item["id"]][0],
                retrieval_ms=prepared[item["id"]][1],
                token_budget=turn_budget,
                lam=lam,
                embedding_model=embedding_model,
                answer_model=answer_model,
                judge_model=judge_model,
                use_jax=use_jax,
                ledger=ledger if _normalize_strategy(strategy) == "mosaic_full" else None,
                cross_turn_lambda=cross_turn_lambda,
                prior_chunks=prior_chunks,
                require_live_answer=require_live_answer,
                require_live_judge=require_live_judge,
            )
            ledger.add_chunks([chunk_lookup[chunk_id] for chunk_id in row["selected_chunk_ids"] if chunk_id in chunk_lookup], turn=int(item.get("turn", 0)), query_id=item["id"], faithfulness=row["faithfulness"])
            unique_tokens = sum(chunk_lookup[chunk_id].token_count for chunk_id in sorted(ledger.selected_ids) if chunk_id in chunk_lookup)
            row.update(
                {
                    "strategy": _normalize_strategy(strategy),
                    "remaining_budget": ledger.remaining_budget,
                    "cumulative_tokens_used": ledger.tokens_used,
                    "unique_tokens_used": unique_tokens,
                    "budget_utilization": round(unique_tokens / max(conversation_budget, 1), 4),
                    "turn_budget": turn_budget,
                }
            )
            rows.append(row)
    if output_path is not None:
        dump_json(output_path, rows)
    return rows


def aggregate_multiturn(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy[row["strategy"]].append(row)

    strategies = []
    per_turn = []
    for strategy, values in sorted(by_strategy.items()):
        scenario_final = {}
        for row in values:
            scenario_id = row["scenario_id"]
            if scenario_id is None or scenario_id not in scenario_final or int(row.get("turn") or 0) > int(scenario_final[scenario_id].get("turn") or 0):
                scenario_final[scenario_id] = row
        strategies.append(
            {
                "strategy": strategy,
                "cross_turn_redundancy": round(mean([row.get("cross_turn_redundancy", 0.0) for row in values if row.get("turn") and int(row["turn"]) > 1]), 4),
                "budget_utilization": round(mean([row.get("budget_utilization", 0.0) for row in scenario_final.values()]), 4),
                "cumulative_faithfulness": round(mean([row["faithfulness"] for row in values]), 4),
            }
        )
        turns = sorted({int(row["turn"]) for row in values if row.get("turn") is not None})
        for turn in turns:
            turn_rows = [row for row in values if int(row.get("turn") or 0) == turn]
            per_turn.append(
                {
                    "strategy": strategy,
                    "turn": turn,
                    "remaining_budget": round(mean([row.get("remaining_budget", 0.0) for row in turn_rows]), 3),
                    "faithfulness": round(mean([row["faithfulness"] for row in turn_rows]), 4),
                    "cross_turn_redundancy": round(mean([row.get("cross_turn_redundancy", 0.0) for row in turn_rows]), 4),
                }
            )
    return {"strategies": strategies, "per_turn": per_turn}


def build_benchmark_summary(single_turn_rows: list[dict[str, Any]], multi_turn_rows: list[dict[str, Any]], failure_summary: dict[str, Any]) -> dict[str, Any]:
    single_summary = aggregate_results(single_turn_rows)
    multi_summary = aggregate_multiturn(multi_turn_rows)
    strategy_lookup = {row["strategy"]: dict(row) for row in single_summary["strategies"]}
    for row in multi_summary["strategies"]:
        strategy_lookup.setdefault(row["strategy"], {"strategy": row["strategy"]})
        strategy_lookup[row["strategy"]].update(row)
    return {
        "strategy_metrics": [strategy_lookup[key] for key in sorted(strategy_lookup)],
        "category_metrics": single_summary["categories"],
        "multi_turn_curves": multi_summary["per_turn"],
        "classification": failure_summary,
    }


def run_benchmark(
    index_path: str | Path,
    eval_suite_path: str | Path,
    token_budget: int = 4096,
    conversation_budget: int = 2200,
    candidate_k: int = 50,
    turn_budget_cap: int | None = None,
    lam: float = 0.5,
    cross_turn_lambda: float = 0.75,
    answer_model: str | None = None,
    judge_model: str | None = None,
    use_jax: bool | None = None,
    output_path: str | Path | None = None,
    pareto_steps: int = 20,
    max_lam: float = 2.0,
    require_live_answer: bool = False,
    require_live_judge: bool = False,
) -> dict[str, Any]:
    index = load_index(index_path)
    embedding_model = str(index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    suite_payload = load_json(eval_suite_path)
    thresholds, calibration_summary, _ = calibrate_thresholds(index.chunks, suite_payload, candidate_k=candidate_k, embedding_model=embedding_model)

    single_turn_rows: list[dict[str, Any]] = []
    for strategy in ("topk", "mmr", "mosaic_no_ledger"):
        single_turn_rows.extend(
            run_eval(
                index_path=index_path,
                eval_suite_path=eval_suite_path,
                strategy=strategy,
                token_budget=token_budget,
                candidate_k=candidate_k,
                lam=lam,
                answer_model=answer_model,
                judge_model=judge_model,
                use_jax=use_jax,
                require_live_answer=require_live_answer,
                require_live_judge=require_live_judge,
            )
        )
    measured_single_turn_rows = list(single_turn_rows)
    single_turn_rows.extend([{**row, "strategy": "mosaic_full"} for row in single_turn_rows if row["strategy"] == "mosaic_no_ledger"])

    failure_rows, failure_summary, resolved_thresholds = run_signal_eval(
        index.chunks,
        suite_payload,
        thresholds=thresholds,
        candidate_k=candidate_k,
        embedding_model=embedding_model,
    )

    multi_turn_rows: list[dict[str, Any]] = []
    for strategy in ("topk", "mmr", "mosaic_no_ledger", "mosaic_full"):
        multi_turn_rows.extend(
            run_multiturn_eval(
                index_path=index_path,
                eval_suite_path=eval_suite_path,
                strategy=strategy,
                conversation_budget=conversation_budget,
                candidate_k=candidate_k,
                lam=lam,
                cross_turn_lambda=cross_turn_lambda,
                answer_model=answer_model,
                judge_model=judge_model,
                use_jax=use_jax,
                require_live_answer=require_live_answer,
                require_live_judge=require_live_judge,
            )
        )

    pareto_rows = run_pareto(
        index_path=index_path,
        eval_suite_path=eval_suite_path,
        token_budget=token_budget,
        candidate_k=candidate_k,
        lam_values=[round((max_lam / max(pareto_steps - 1, 1)) * idx, 4) for idx in range(pareto_steps)],
        answer_model=answer_model,
        judge_model=judge_model,
        use_jax=use_jax,
        require_live_answer=require_live_answer,
        require_live_judge=require_live_judge,
    )

    evaluation_provenance = summarize_eval_provenance(
        measured_single_turn_rows + multi_turn_rows,
        answer_model=answer_model,
        judge_model=judge_model,
        require_live_answer=require_live_answer,
        require_live_judge=require_live_judge,
    )
    payload = {
        "metadata": {
            "token_budget": token_budget,
            "conversation_budget": conversation_budget,
            "candidate_k": candidate_k,
            "turn_budget_cap": turn_budget_cap if turn_budget_cap is not None else max(384, conversation_budget // 4),
            "lam": lam,
            "cross_turn_lambda": cross_turn_lambda,
            "answer_model": answer_model or DEFAULT_ANTHROPIC_ANSWER_MODEL,
            "judge_model": judge_model or DEFAULT_ANTHROPIC_JUDGE_MODEL,
            "thresholds": resolved_thresholds.to_dict() if isinstance(resolved_thresholds, SignalThresholds) else SignalThresholds().to_dict(),
            "calibration_summary": calibration_summary,
            "evaluation_provenance": evaluation_provenance,
        },
        "single_turn_rows": single_turn_rows,
        "failure_rows": failure_rows,
        "multi_turn_rows": multi_turn_rows,
        "pareto_rows": pareto_rows,
        "summary": build_benchmark_summary(single_turn_rows, multi_turn_rows, failure_summary),
    }
    if output_path is not None:
        dump_json(output_path, payload)
    return payload
