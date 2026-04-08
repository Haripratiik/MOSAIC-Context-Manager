# MOSAIC

Minimum Overlap Semantic Assembly with Information Coverage.

MOSAIC is a context-management layer for retrieval-heavy AI systems. It sits between a corpus and an AI caller, selects the best permitted snippets under a token budget, explains failure modes, and records an auditable trace of what happened.

## What It Does

- Optimizes context selection instead of relying on naive top-k ranking
- Penalizes redundant chunks so the budget is spent on new information
- Enforces per-document or per-chunk permission labels
- Distinguishes `SUCCESS`, `TRUE_UNKNOWN`, `PERMISSION_GAP`, and `RETRIEVAL_FAILURE`
- Tracks cross-turn budget and reuse through a conversation ledger
- Exposes the same runtime through CLI, browser UI, and MCP
- Stores audit traces in SQLite and exports them as JSON

## Main Surfaces

- CLI: ingestion, querying, benchmarking, reporting, perf, audit, MCP, UI
- Browser UI: query, corpus browser, audit trace, workspace health, report view
- MCP: `stdio` and Streamable HTTP transports
- Offline report: Plotly dashboard for optimizer, classifier, multi-turn, governance, and perf evidence

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
.\.venv\Scripts\mosaic.exe demo --clean --fast
```

That run builds a local workspace in `.mosaic/` with:

- `index.json`
- `benchmark.json`
- `pareto.json`
- `perf.json`
- `report.html`
- `audit.db`

Then check the workspace:

```powershell
.\.venv\Scripts\mosaic.exe doctor --strict
```

## First Live Query

```powershell
.\.venv\Scripts\mosaic.exe query `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --query "Which recurring market-risk factors are repeated across the filings for Acadia Securities?" `
  --roles senior_analyst `
  --strategy mosaic_full `
  --conversation-budget 900
```

Look for:

- `status`
- `classification`
- `selected_chunks`
- `audit_id`
- `conversation_id`
- `remaining_budget`

Use that `audit_id` to inspect the trace:

```powershell
.\.venv\Scripts\mosaic.exe audit show `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --audit-id <audit_id> `
  --summary `
  --candidate-limit 10
```

## Run The UI

```powershell
.\.venv\Scripts\mosaic.exe ui `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --report-path .mosaic/report.html `
  --benchmark-path .mosaic/benchmark.json `
  --perf-path .mosaic/perf.json `
  --open-ui
```

The UI gives you:

- `Query`: the live answer surface
- `Corpus`: the loaded source documents
- `Audit`: the explanation trace
- `Workspace`: artifact and backend readiness
- `Report`: the offline dashboard

## Bring Your Own Corpus

MOSAIC supports direct ingestion of:

- `.md`
- `.markdown`
- `.txt`

Fastest own-data flow:

```powershell
.\.venv\Scripts\mosaic.exe ingest my_docs --output .mosaic/my_index.json --embedding-model hash --vector-store json
.\.venv\Scripts\mosaic.exe query --index .mosaic/my_index.json --query "Your question here" --roles public --strategy mosaic_no_ledger --budget 700
.\.venv\Scripts\mosaic.exe ui --index .mosaic/my_index.json --open-ui
```

You can also ingest via manifest for explicit titles, IDs, permission labels, and metadata:

```powershell
.\.venv\Scripts\mosaic.exe ingest `
  --manifest examples/project_docs_manifest.json `
  --output .mosaic/project_docs_index.json `
  --embedding-model hash `
  --vector-store json
```

## MCP

Local `stdio`:

```powershell
.\.venv\Scripts\mosaic.exe serve-mcp `
  --transport stdio `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db
```

Streamable HTTP:

```powershell
.\.venv\Scripts\mosaic.exe serve-mcp `
  --transport http `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --principal-map examples/principals.sample.json `
  --allow-origin http://localhost
```

## Optional Live Evaluation

The core runtime works offline. If `ANTHROPIC_API_KEY` is set, MOSAIC can also use live Anthropic-backed evaluation paths for answer and judge measurements. When no key is available, the report and benchmark record explicit proxy provenance instead of silently pretending those measurements were live.

## Documentation Map

- [HOW_TO_RUN.md](HOW_TO_RUN.md): practical runbook
- [docs/QUICKSTART.md](docs/QUICKSTART.md): shortest path from clone to working system
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): runtime flow and module responsibilities
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): extension guide
- [examples/README.md](examples/README.md): manifests and principal maps

## Repository Layout

- `mosaic/`: runtime package
- `tests/`: unit and integration tests
- `corpus/`: corpus and eval generators
- `examples/`: manifest and auth samples
- `docs/`: usage and architecture notes

Generated artifacts stay out of the source tree and land in `.mosaic/`.

## Testing

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Project Model

MOSAIC is best thought of as middleware:

1. ingest a corpus
2. send a query plus permission labels
3. let MOSAIC assemble the best permitted context under a budget
4. return the answer package and an audit trace for downstream use

That same runtime is shared by the CLI, UI, MCP server, benchmark flow, and report generator.
