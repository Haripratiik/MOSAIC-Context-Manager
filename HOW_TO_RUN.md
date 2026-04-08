# How To Run MOSAIC

This is the practical runbook for using MOSAIC locally.

## Requirements

- Python 3.11+
- PowerShell or an equivalent shell
- Internet access is optional
- `ANTHROPIC_API_KEY` is optional

Without `ANTHROPIC_API_KEY`, the core runtime still works end to end. Live evaluation paths simply fall back to explicit proxy provenance.

## Fastest End-To-End Demo

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
.\.venv\Scripts\mosaic.exe demo --clean --fast
```

This creates the main local artifacts in `.mosaic/`:

- `index.json`
- `benchmark.json`
- `pareto.json`
- `perf.json`
- `report.html`
- `audit.db`

Use `--no-open` for headless runs and `--open-ui` if you want the browser UI to launch automatically after the build.

## Validate The Workspace

```powershell
.\.venv\Scripts\mosaic.exe doctor --strict
```

Healthy output should report `ok: true`.

## Ask One Live Query

```powershell
.\.venv\Scripts\mosaic.exe query `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --query "Which recurring market-risk factors are repeated across the filings for Acadia Securities?" `
  --roles senior_analyst `
  --strategy mosaic_full `
  --conversation-budget 900
```

Key fields in the result:

- `status`
- `classification`
- `selected_chunks`
- `audit_id`
- `conversation_id`
- `remaining_budget`

## Inspect The Audit Trace

```powershell
.\.venv\Scripts\mosaic.exe audit show `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --audit-id <audit_id> `
  --summary `
  --candidate-limit 10
```

This is the fastest way to see:

- selected chunks
- denied candidates
- non-selection factor lists
- timing summaries
- conversation state

## Launch The Browser UI

```powershell
.\.venv\Scripts\mosaic.exe ui `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --report-path .mosaic/report.html `
  --benchmark-path .mosaic/benchmark.json `
  --perf-path .mosaic/perf.json `
  --open-ui
```

UI panes:

- `Query`: answer output
- `Corpus`: loaded source documents
- `Audit`: explanation trace
- `Workspace`: readiness and health checks
- `Report`: offline dashboard

## Use MOSAIC On Your Own Data

Supported direct source types:

- `.md`
- `.markdown`
- `.txt`

Fast path:

```powershell
.\.venv\Scripts\mosaic.exe ingest my_docs --output .mosaic/my_index.json --embedding-model hash --vector-store json
.\.venv\Scripts\mosaic.exe query --index .mosaic/my_index.json --query "Your question here" --roles public --strategy mosaic_no_ledger --budget 700
.\.venv\Scripts\mosaic.exe ui --index .mosaic/my_index.json --open-ui
```

Permission labels are corpus-defined. The demo uses labels like `analyst` and `compliance`, but your own corpus can use labels like `employee`, `finance`, `legal`, `reader`, or `*`.

## Manifest-Based Ingestion

Use a manifest when you want explicit document IDs, titles, permission labels, or metadata:

```powershell
.\.venv\Scripts\mosaic.exe ingest `
  --manifest examples/project_docs_manifest.json `
  --output .mosaic/project_docs_index.json `
  --embedding-model hash `
  --vector-store json
```

Then query it:

```powershell
.\.venv\Scripts\mosaic.exe query `
  --index .mosaic/project_docs_index.json `
  --query "What does the README say about MCP?" `
  --roles public `
  --strategy mosaic_no_ledger `
  --budget 700
```

## Serve MOSAIC Over MCP

### stdio

```powershell
.\.venv\Scripts\mosaic.exe serve-mcp `
  --transport stdio `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db
```

### HTTP

```powershell
.\.venv\Scripts\mosaic.exe serve-mcp `
  --transport http `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --principal-map examples/principals.sample.json `
  --allow-origin http://localhost
```

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Practical Reading Order

1. [README.md](README.md)
2. [docs/QUICKSTART.md](docs/QUICKSTART.md)
3. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
4. [examples/README.md](examples/README.md)
