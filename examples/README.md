# Examples

This folder contains ready-to-use inputs for the real-corpus and MCP flows.

## Files

- [`project_docs_manifest.json`](project_docs_manifest.json): a manifest that indexes a small set of repository docs for real-corpus testing
- [`principals.sample.json`](principals.sample.json): demo principals and bearer tokens that match the bundled finserv corpus
- [`principals.generic.sample.json`](principals.generic.sample.json): a neutral principal map with reusable labels like `member` and `*`

## Typical Uses

### Ingest The Repository Docs

```powershell
.\.venv\Scripts\mosaic.exe ingest `
  --manifest examples/project_docs_manifest.json `
  --output .mosaic/project_docs_index.json `
  --embedding-model hash `
  --vector-store json
```

### Query The Repository Docs

```powershell
.\.venv\Scripts\mosaic.exe query `
  --index .mosaic/project_docs_index.json `
  --query "What does the README say about MCP?" `
  --roles public `
  --strategy mosaic_no_ledger `
  --budget 700
```

### Start HTTP MCP With Sample Principals

```powershell
.\.venv\Scripts\mosaic.exe serve-mcp `
  --transport http `
  --index .mosaic/index.json `
  --audit-db .mosaic/audit.db `
  --principal-map examples/principals.sample.json `
  --allow-origin http://localhost
```
