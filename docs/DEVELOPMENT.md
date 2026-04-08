# Development Guide

This document is for someone changing or extending the code rather than only using it.

## Working Style

The codebase is easiest to work with if you keep one rule in mind:

- public surfaces stay thin
- runtime behavior stays centralized

In practice, that means:

- CLI commands should call [`mosaic/service.py`](../mosaic/service.py) or a focused helper module
- the MCP server should expose the same runtime, not a second code path
- the UI should read the same artifacts the CLI produces

## Common Change Paths

### Change Retrieval Behavior

Start with:

- [`mosaic/retriever.py`](../mosaic/retriever.py)
- [`mosaic/ingestor.py`](../mosaic/ingestor.py)
- [`mosaic/service.py`](../mosaic/service.py)

### Change The Optimizer

Start with:

- [`mosaic/optimizer.py`](../mosaic/optimizer.py)
- [`mosaic/ledger.py`](../mosaic/ledger.py)
- [`tests/test_optimizer.py`](../tests/test_optimizer.py)
- [`tests/test_ledger.py`](../tests/test_ledger.py)

### Change Failure Classification

Start with:

- [`mosaic/signal.py`](../mosaic/signal.py)
- [`tests/test_signal.py`](../tests/test_signal.py)

### Change MCP Or Auth

Start with:

- [`mosaic/mcp_server.py`](../mosaic/mcp_server.py)
- [`mosaic/auth.py`](../mosaic/auth.py)
- [`tests/test_mcp.py`](../tests/test_mcp.py)
- [`tests/test_auth.py`](../tests/test_auth.py)

### Change Audit Or Governance

Start with:

- [`mosaic/audit.py`](../mosaic/audit.py)
- [`mosaic/report.py`](../mosaic/report.py)
- [`tests/test_audit_cli.py`](../tests/test_audit_cli.py)

### Change The UI Or Workspace Checks

Start with:

- [`mosaic/ui.py`](../mosaic/ui.py)
- [`mosaic/workspace.py`](../mosaic/workspace.py)
- [`tests/test_ui.py`](../tests/test_ui.py)
- [`tests/test_doctor_cli.py`](../tests/test_doctor_cli.py)

## Fast Verification Commands

### Full Test Suite

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### High-Value Smoke Commands

```powershell
.\.venv\Scripts\mosaic.exe demo --clean --no-open
.\.venv\Scripts\mosaic.exe doctor --strict
.\.venv\Scripts\mosaic.exe ui --help
.\.venv\Scripts\mosaic.exe serve-mcp --help
```

### Real-Corpus Path

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

## Definition Of Done For Changes

Before considering a change complete, try to preserve all of these:

- the CLI path still works
- the same behavior remains available through `MosaicService`
- audit traces still record the important reasoning and permission information
- UI and report still read the generated artifacts cleanly
- tests or smoke checks cover the changed surface

## Notes On External Backends

Some evaluation features can use Anthropic-backed live measurements. The repo also supports explicit proxy fallback so local work stays functional offline.

If you need strict live-only behavior, use:

- `--require-live-answer`
- `--require-live-judge`
