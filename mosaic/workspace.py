from __future__ import annotations

"""Shared workspace and artifact readiness summaries for CLI and UI."""

from pathlib import Path
from typing import Any

from .audit import SQLiteAuditStore
from .auth import PrincipalResolver
from .evaluator import backend_capabilities
from .ingestor import load_index
from .utils import DEFAULT_EMBEDDING_MODEL, load_json


def _benchmark_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'path': str(path), 'exists': False}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {'path': str(path), 'exists': True, 'valid': False}
    summary = payload.get('summary', {}) if isinstance(payload.get('summary'), dict) else {}
    metadata = payload.get('metadata', {}) if isinstance(payload.get('metadata'), dict) else {}
    return {
        'path': str(path),
        'exists': True,
        'valid': True,
        'strategy_count': len(summary.get('strategy_metrics', [])),
        'single_turn_rows': len(payload.get('single_turn_rows', [])),
        'failure_rows': len(payload.get('failure_rows', [])),
        'multi_turn_rows': len(payload.get('multi_turn_rows', [])),
        'evaluation_provenance': metadata.get('evaluation_provenance'),
    }


def _perf_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'path': str(path), 'exists': False}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {'path': str(path), 'exists': True, 'valid': False}
    summary = payload.get('summary', {}) if isinstance(payload.get('summary'), dict) else {}
    metadata = payload.get('metadata', {}) if isinstance(payload.get('metadata'), dict) else {}
    return {
        'path': str(path),
        'exists': True,
        'valid': True,
        'strategy': metadata.get('strategy'),
        'request_count': summary.get('request_count', 0),
        'throughput_qps': summary.get('throughput_qps', 0.0),
        'latency_ms': summary.get('latency_ms', {}),
        'status_counts': summary.get('status_counts', []),
    }


def build_workspace_summary(
    *,
    index_path: str | Path,
    audit_db_path: str | Path,
    principal_map_path: str | Path,
    report_path: str | Path,
    answer_model: str,
    judge_model: str,
    strict: bool = False,
    require_live_answer: bool = False,
    require_live_judge: bool = False,
    benchmark_path: str | Path | None = None,
    perf_path: str | Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'ok': True,
        'strict': strict,
        'checks': {},
        'warnings': [],
    }

    resolved_index_path = Path(index_path)
    index_row = {
        'path': str(resolved_index_path),
        'exists': resolved_index_path.exists(),
    }
    if resolved_index_path.exists():
        index = load_index(resolved_index_path)
        index_row.update(
            {
                'chunks': len(index.chunks),
                'documents': len({chunk.document_id for chunk in index.chunks}),
                'embedding_model': index.config.get('embedding_model', DEFAULT_EMBEDDING_MODEL),
                'vector_store': index.config.get('vector_store', 'json'),
                'corpus_source': index.config.get('corpus_source', 'directory'),
                'manifest_path': index.config.get('manifest_path'),
            }
        )
    else:
        payload['ok'] = False
        payload['warnings'].append('Index is missing. Run `mosaic demo --clean` or `mosaic ingest` first.')
    payload['checks']['index'] = index_row

    resolved_audit_path = Path(audit_db_path)
    audit_row = {
        'path': str(resolved_audit_path),
        'exists': resolved_audit_path.exists(),
    }
    if resolved_audit_path.exists():
        audit_row.update(SQLiteAuditStore(resolved_audit_path).stats())
    elif strict:
        payload['ok'] = False
        payload['warnings'].append('Audit DB is missing in strict mode.')
    payload['checks']['audit_db'] = audit_row

    resolved_principal_map = Path(principal_map_path)
    principal_row = {
        'path': str(resolved_principal_map),
        'exists': resolved_principal_map.exists(),
    }
    if resolved_principal_map.exists():
        resolver = PrincipalResolver.from_file(resolved_principal_map)
        principal_row.update(
            {
                'principal_count': len(resolver.principals),
                'token_count': len(resolver.tokens),
                'http_principals': sorted(
                    principal.principal_id
                    for principal in resolver.principals.values()
                    if principal.transport == 'http'
                ),
            }
        )
    elif strict:
        payload['ok'] = False
        payload['warnings'].append('Principal map is missing in strict mode.')
    payload['checks']['principal_map'] = principal_row

    resolved_report_path = Path(report_path)
    report_row = {
        'path': str(resolved_report_path),
        'exists': resolved_report_path.exists(),
    }
    if resolved_report_path.exists():
        report_row['size_bytes'] = resolved_report_path.stat().st_size
    elif strict:
        payload['ok'] = False
        payload['warnings'].append('Report HTML is missing in strict mode.')
    payload['checks']['report'] = report_row

    live_eval_row = backend_capabilities(answer_model=answer_model, judge_model=judge_model)
    payload['checks']['live_eval'] = live_eval_row
    if require_live_answer and not live_eval_row.get('live_answer_ready'):
        payload['ok'] = False
        payload['warnings'].append('Live answer backend is not ready.')
    if require_live_judge and not live_eval_row.get('live_judge_ready'):
        payload['ok'] = False
        payload['warnings'].append('Live judge backend is not ready.')

    if benchmark_path is not None:
        payload['checks']['benchmark'] = _benchmark_summary(Path(benchmark_path))
    if perf_path is not None:
        payload['checks']['perf'] = _perf_summary(Path(perf_path))

    return payload
