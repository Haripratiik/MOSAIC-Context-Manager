"""Public package surface for MOSAIC's runtime, control plane, and tooling."""

from .auth import AuthError, AuthProvider, BearerTokenAuthProvider, Principal, PrincipalResolver
from .audit import AuditTrace, JsonLedgerStore, SQLiteAuditStore, SQLiteLedgerStore
from .corpus_builder import generate_corpus, generate_eval_suite
from .evaluator import aggregate_results, aggregate_multiturn, backend_capabilities, run_benchmark, run_eval, run_multiturn_eval, run_pareto, summarize_eval_provenance
from .ledger import ContextLedger, LedgerStore
from .mcp_server import build_http_app, build_mcp_server, serve_mcp
from .perf import run_perf
from .ui import build_ui_app, serve_ui
from .optimizer import OptimizationResult, optimize, pareto_sweep, select_mmr, select_topk
from .service import MosaicService, QueryRequest, QueryResponse
from .signal import SignalThresholds, classify_query, run_signal_eval
from .types import Chunk, RetrievalIndex
from .workspace import build_workspace_summary

__all__ = [
    "AuditTrace",
    "AuthError",
    "AuthProvider",
    "BearerTokenAuthProvider",
    "Chunk",
    "ContextLedger",
    "JsonLedgerStore",
    "LedgerStore",
    "MosaicService",
    "OptimizationResult",
    "Principal",
    "PrincipalResolver",
    "QueryRequest",
    "QueryResponse",
    "RetrievalIndex",
    "SQLiteAuditStore",
    "SQLiteLedgerStore",
    "SignalThresholds",
    "aggregate_multiturn",
    "aggregate_results",
    "backend_capabilities",
    "build_http_app",
    "build_mcp_server",
    "build_ui_app",
    "classify_query",
    "generate_corpus",
    "generate_eval_suite",
    "optimize",
    "pareto_sweep",
    "run_benchmark",
    "run_eval",
    "run_multiturn_eval",
    "run_pareto",
    "run_perf",
    "serve_ui",
    "run_signal_eval",
    "summarize_eval_provenance",
    "select_mmr",
    "select_topk",
    "build_workspace_summary",
    "serve_mcp",
]

__version__ = "0.7.0"
