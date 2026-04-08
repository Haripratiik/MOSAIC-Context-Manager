from __future__ import annotations

"""Runtime performance harness for throughput and latency evidence."""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .audit import SQLiteAuditStore, SQLiteLedgerStore
from .auth import PrincipalResolver
from .service import MosaicService, QueryRequest
from .utils import dump_json, load_json


def _suite_queries(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and 'queries' in payload:
        return list(payload['queries'])
    return list(payload)  # type: ignore[arg-type]


def _normalize_query_items(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        query = str(item.get('query', '')).strip()
        if not query:
            continue
        roles = item.get('user_roles') or item.get('roles') or ['public']
        if isinstance(roles, str):
            roles = [part.strip() for part in roles.split(',') if part.strip()]
        normalized.append(
            {
                'id': str(item.get('id') or f'perf-query-{index + 1}'),
                'query': query,
                'roles': list(roles),
                'category': item.get('category', 'adhoc'),
                'turn': item.get('turn'),
                'scenario_id': item.get('scenario_id'),
            }
        )
    return normalized


def load_perf_queries(
    *,
    suite_path: str | Path | None = None,
    queries: list[dict[str, Any]] | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
    include_multi_turn: bool = False,
) -> list[dict[str, Any]]:
    raw_items: list[dict[str, Any]]
    if queries is not None:
        raw_items = list(queries)
    elif suite_path is not None:
        payload = load_json(suite_path)
        raw_items = list(_suite_queries(payload))
    else:
        raise ValueError('Provide either suite_path or queries for perf testing.')

    normalized = _normalize_query_items(raw_items)
    if categories:
        allowed = {category.strip() for category in categories if category.strip()}
        normalized = [item for item in normalized if item['category'] in allowed]
    if not include_multi_turn:
        normalized = [item for item in normalized if item['category'] != 'multi_turn']
    if limit is not None:
        normalized = normalized[:limit]
    if not normalized:
        raise ValueError('No perf queries matched the requested filters.')
    return normalized


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}
    return {
        'mean': round(sum(values) / len(values), 3),
        'p50': round(_percentile(values, 0.50), 3),
        'p95': round(_percentile(values, 0.95), 3),
        'p99': round(_percentile(values, 0.99), 3),
        'max': round(max(values), 3),
    }


def _execute_request(
    service: MosaicService,
    item: dict[str, Any],
    *,
    strategy: str,
    token_budget: int,
    candidate_k: int,
    lam: float,
    cross_turn_lambda: float,
    request_suffix: str,
) -> dict[str, Any]:
    response = service.query(
        QueryRequest(
            query=item['query'],
            requested_roles=list(item.get('roles', ['public'])),
            strategy=strategy,
            candidate_k=candidate_k,
            token_budget=token_budget,
            conversation_id=f"perf-{item['id']}-{request_suffix}" if strategy == 'mosaic_full' else None,
            turn=int(item['turn']) if item.get('turn') is not None else None,
            lam=lam,
            cross_turn_lambda=cross_turn_lambda,
        ),
        transport='perf',
    )
    return {
        'query_id': item['id'],
        'query': item['query'],
        'category': item.get('category', 'adhoc'),
        'status': response.status,
        'classification': response.classification,
        'tokens_used': response.tokens_used,
        'remaining_budget': response.remaining_budget,
        'selected_chunk_count': len(response.selected_chunk_ids),
        'timings_ms': dict(response.timings_ms),
        'audit_id': response.audit_id,
        'conversation_id': response.conversation_id,
    }


def run_perf(
    index_path: str | Path,
    *,
    suite_path: str | Path | None = None,
    queries: list[dict[str, Any]] | None = None,
    strategy: str = 'mosaic_no_ledger',
    token_budget: int = 4096,
    conversation_budget: int = 2200,
    candidate_k: int = 50,
    lam: float = 0.5,
    cross_turn_lambda: float = 0.75,
    iterations: int = 1,
    warmup: int = 0,
    concurrency: int = 1,
    limit: int | None = None,
    categories: list[str] | None = None,
    include_multi_turn: bool = False,
    audit_db_path: str | Path | None = None,
    principal_map_path: str | Path | None = None,
    calibration_suite_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError('iterations must be positive.')
    if warmup < 0:
        raise ValueError('warmup must be non-negative.')
    if concurrency <= 0:
        raise ValueError('concurrency must be positive.')
    if candidate_k <= 0:
        raise ValueError('candidate_k must be positive.')

    query_items = load_perf_queries(
        suite_path=suite_path,
        queries=queries,
        categories=categories,
        limit=limit,
        include_multi_turn=include_multi_turn,
    )
    total_budget = conversation_budget if strategy == 'mosaic_full' else token_budget
    resolver = PrincipalResolver.from_file(principal_map_path)
    service = MosaicService(
        index_path,
        audit_store=SQLiteAuditStore(audit_db_path) if audit_db_path else None,
        ledger_store=SQLiteLedgerStore(audit_db_path) if audit_db_path else None,
        principal_resolver=resolver,
        calibration_suite_path=calibration_suite_path,
    )

    work_items: list[tuple[str, dict[str, Any]]] = []
    for iteration in range(iterations):
        for item in query_items:
            work_items.append((f'{iteration + 1}-{item["id"]}', item))

    warmup_count = min(warmup, len(work_items))
    warmup_items = work_items[:warmup_count]
    measured_items = work_items[warmup_count:]
    if not measured_items:
        raise ValueError('Warmup consumed all perf requests; reduce --warmup or increase the query count.')

    for request_suffix, item in warmup_items:
        _execute_request(
            service,
            item,
            strategy=strategy,
            token_budget=total_budget,
            candidate_k=candidate_k,
            lam=lam,
            cross_turn_lambda=cross_turn_lambda,
            request_suffix=request_suffix,
        )

    started = time.perf_counter()
    if concurrency == 1:
        rows = [
            _execute_request(
                service,
                item,
                strategy=strategy,
                token_budget=total_budget,
                candidate_k=candidate_k,
                lam=lam,
                cross_turn_lambda=cross_turn_lambda,
                request_suffix=request_suffix,
            )
            for request_suffix, item in measured_items
        ]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _execute_request,
                    service,
                    item,
                    strategy=strategy,
                    token_budget=total_budget,
                    candidate_k=candidate_k,
                    lam=lam,
                    cross_turn_lambda=cross_turn_lambda,
                    request_suffix=request_suffix,
                )
                for request_suffix, item in measured_items
            ]
            rows = [future.result() for future in futures]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    status_counts: dict[str, int] = defaultdict(int)
    classification_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[str(row['status'])] += 1
        classification_counts[str(row['classification'])] += 1

    latency_summary = {
        'total': _metric_summary([float(row['timings_ms'].get('total', 0.0)) for row in rows]),
        'classification': _metric_summary([float(row['timings_ms'].get('classification', 0.0)) for row in rows]),
        'retrieval': _metric_summary([float(row['timings_ms'].get('retrieval', 0.0)) for row in rows]),
        'optimization': _metric_summary([float(row['timings_ms'].get('optimization', 0.0)) for row in rows]),
        'answer': _metric_summary([float(row['timings_ms'].get('answer', 0.0)) for row in rows]),
    }

    payload = {
        'metadata': {
            'strategy': strategy,
            'token_budget': token_budget,
            'conversation_budget': conversation_budget,
            'effective_token_budget': total_budget,
            'candidate_k': candidate_k,
            'lam': lam,
            'cross_turn_lambda': cross_turn_lambda,
            'iterations': iterations,
            'warmup': warmup_count,
            'concurrency': concurrency,
            'query_count': len(query_items),
            'request_count': len(rows),
            'categories': sorted({item['category'] for item in query_items}),
            'include_multi_turn': include_multi_turn,
            'audit_enabled': audit_db_path is not None,
            'suite_path': str(suite_path) if suite_path is not None else None,
            'principal_map_path': str(principal_map_path) if principal_map_path is not None else None,
        },
        'summary': {
            'request_count': len(rows),
            'elapsed_ms': round(elapsed_ms, 3),
            'throughput_qps': round((len(rows) / elapsed_ms) * 1000.0, 3) if elapsed_ms > 0 else 0.0,
            'status_counts': [
                {'status': status, 'count': count}
                for status, count in sorted(status_counts.items())
            ],
            'classification_counts': [
                {'classification': classification, 'count': count}
                for classification, count in sorted(classification_counts.items())
            ],
            'latency_ms': latency_summary,
            'tokens_used': _metric_summary([float(row['tokens_used']) for row in rows]),
            'selected_chunk_count': _metric_summary([float(row['selected_chunk_count']) for row in rows]),
            'remaining_budget': _metric_summary([
                float(row['remaining_budget'])
                for row in rows
                if row.get('remaining_budget') is not None
            ]),
        },
        'samples': rows[: min(10, len(rows))],
    }
    if output_path is not None:
        dump_json(output_path, payload)
    return payload
