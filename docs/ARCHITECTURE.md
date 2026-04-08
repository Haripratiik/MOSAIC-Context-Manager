# Architecture

This document explains how the main MOSAIC pieces fit together.

## Mental Model

MOSAIC is a context-management layer, not just a retriever.

It combines:

- an assembler that optimizes chunk selection under a hard budget
- a classifier that explains why a query could not be served
- a cross-turn ledger that avoids spending the same budget twice
- a control plane that exposes the runtime over MCP and records an audit trail

## Main Surfaces

### CLI

[`mosaic/cli.py`](../mosaic/cli.py) is the operator-facing surface. It handles demo runs, ingestion, querying, auditing, benchmarking, reporting, MCP serving, perf runs, and the browser UI.

### Canonical Runtime

[`mosaic/service.py`](../mosaic/service.py) is the single orchestration layer. The CLI, MCP server, and browser UI all call into this service so the behavior stays consistent.

### MCP Control Plane

[`mosaic/mcp_server.py`](../mosaic/mcp_server.py) exposes the runtime over:

- `stdio`
- Streamable HTTP

The HTTP path adds origin checks and bearer-token authentication.

### Audit And Ledger Store

[`mosaic/audit.py`](../mosaic/audit.py) owns:

- SQLite audit persistence
- JSON export
- the SQLite-backed conversation ledger
- governance summaries used by the report

### Report And UI

[`mosaic/report.py`](../mosaic/report.py) renders the offline HTML dashboard.

[`mosaic/ui.py`](../mosaic/ui.py) serves the local browser control plane for query, audit, workspace, and report workflows.

## Middleware View

In standalone use, MOSAIC is the filter/context layer between raw documents and an AI caller. The corpus is ingested once, then CLI, UI, or MCP callers send a question plus permission labels into the canonical runtime. The runtime returns a structured response and audit trace instead of forcing every caller to reimplement retrieval, filtering, and failure handling.

## Request Lifecycle

For a normal query, the flow is:

1. The caller submits a `QueryRequest` through the CLI, MCP, or UI.
2. `MosaicService` resolves the principal and effective roles.
3. The classifier assigns one of:
   - `SUCCESS`
   - `TRUE_UNKNOWN`
   - `PERMISSION_GAP`
   - `RETRIEVAL_FAILURE`
4. If the query is answerable, the retriever plus selected strategy build the context:
   - `topk`
   - `mmr`
   - `mosaic_no_ledger`
   - `mosaic_full`
5. If ledger mode is active, the conversation budget and prior-context state are updated.
6. The runtime generates a response, builds an `AuditTrace`, and writes it to SQLite when an audit DB is configured.
7. The caller receives a `QueryResponse` with `audit_id` and, when applicable, `conversation_id`.

## Module Guide

| Module | Responsibility |
|---|---|
| `mosaic/service.py` | canonical runtime orchestration |
| `mosaic/retriever.py` | hybrid retrieval and candidate assembly |
| `mosaic/optimizer.py` | MOSAIC optimization and baseline strategies |
| `mosaic/signal.py` | failure classification and threshold calibration |
| `mosaic/ledger.py` | in-memory conversation ledger behavior |
| `mosaic/audit.py` | SQLite audit store and SQLite ledger persistence |
| `mosaic/auth.py` | principals, bearer auth, and role resolution |
| `mosaic/mcp_server.py` | MCP tool and resource surface |
| `mosaic/evaluator.py` | eval, benchmark, and provenance handling |
| `mosaic/perf.py` | perf and load evidence generation |
| `mosaic/report.py` | HTML report rendering |
| `mosaic/ui.py` | local browser control plane |
| `mosaic/workspace.py` | workspace readiness summaries |

## Generated Artifacts

Most generated output lives in `.mosaic/`:

- `index.json`
- `benchmark.json`
- `pareto.json`
- `perf.json`
- `report.html`
- `audit.db`

## Extension Points

Common swaps are localized:

- new auth provider: extend [`mosaic/auth.py`](../mosaic/auth.py)
- new runtime surface: call `MosaicService` instead of adding a parallel code path
- new evaluation backend: extend [`mosaic/evaluator.py`](../mosaic/evaluator.py)
- new report section: extend [`mosaic/report.py`](../mosaic/report.py)
- new corpus source: extend [`mosaic/ingestor.py`](../mosaic/ingestor.py)

The main architectural rule is simple: keep orchestration in `MosaicService` and keep surfaces thin.
