# Quickstart

This is the shortest path from clone to a working MOSAIC setup.

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
```

## 2. Run The Full Demo

```powershell
.\.venv\Scripts\mosaic.exe demo --clean --fast
```

That produces:

- `.mosaic/index.json`
- `.mosaic/benchmark.json`
- `.mosaic/pareto.json`
- `.mosaic/perf.json`
- `.mosaic/report.html`
- `.mosaic/audit.db`

Use `--no-open` for headless runs and `--open-ui` if you want the browser UI to launch automatically.

## 3. Ask One Question

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

## 4. Inspect The Audit Trace

```powershell
.\.venv\Scripts\mosaic.exe audit show `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --audit-id <audit_id> `
  --summary `
  --candidate-limit 10
```

## 5. Open The UI

```powershell
.\.venv\Scripts\mosaic.exe ui `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --report-path .mosaic/report.html `
  --benchmark-path .mosaic/benchmark.json `
  --perf-path .mosaic/perf.json `
  --open-ui
```

## 6. Try A Real Corpus

```powershell
.\.venv\Scripts\mosaic.exe ingest `
  --manifest examples/project_docs_manifest.json `
  --output .mosaic/project_docs_index.json `
  --embedding-model hash `
  --vector-store json

.\.venv\Scripts\mosaic.exe query `
  --index .mosaic/project_docs_index.json `
  --query "What does the README say about MCP?" `
  --roles public `
  --strategy mosaic_no_ledger `
  --budget 700
```

## 7. Serve MOSAIC Over MCP

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

## 8. Validate The Workspace

```powershell
.\.venv\Scripts\mosaic.exe doctor --strict
```

## Where To Read Next

1. [README.md](../README.md)
2. [docs/ARCHITECTURE.md](ARCHITECTURE.md)
3. [docs/DEVELOPMENT.md](DEVELOPMENT.md)
4. [examples/README.md](../examples/README.md)
