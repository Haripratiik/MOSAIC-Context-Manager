from __future__ import annotations

"""Local browser UI for querying, auditing, and workspace inspection."""

import json
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
import uvicorn

from .audit import SQLiteAuditStore, SQLiteLedgerStore
from .auth import PrincipalResolver
from .service import MosaicService, QueryRequest
from .types import Chunk
from .workspace import build_workspace_summary


UI_HTML = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>MOSAIC Control Plane</title>
<style>
:root {
  --ink: #10273a;
  --muted: #4f6777;
  --line: #d7e4ea;
  --card: rgba(255,255,255,0.92);
  --accent: #0f766e;
  --accent-2: #c96b1f;
  --bg-1: #f6fbff;
  --bg-2: #edf6ef;
  --bg-3: #fff7ed;
  --shadow: 0 18px 45px rgba(16, 39, 58, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  font-family: Bahnschrift, "Segoe UI Variable", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(255,255,255,0.95) 0%, rgba(246,251,255,0.97) 28%, rgba(237,246,239,0.95) 62%, rgba(255,247,237,0.95) 100%);
}
main { max-width: 1320px; margin: 0 auto; padding: 32px 24px 56px; }
header {
  display: grid;
  gap: 12px;
  margin-bottom: 24px;
}
.hero {
  display: grid;
  gap: 10px;
  padding: 24px;
  border: 1px solid var(--line);
  border-radius: 28px;
  background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,248,239,0.92));
  box-shadow: var(--shadow);
}
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.74rem;
  color: var(--accent);
  font-weight: 700;
}
.hero h1 {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(2.1rem, 4vw, 4rem);
  letter-spacing: -0.04em;
}
.hero p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
  max-width: 74ch;
}
.meta-grid, .card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
}
.meta-card, .card, .panel, .result-block, .list-shell {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: var(--card);
  box-shadow: var(--shadow);
}
.meta-card, .card { padding: 16px; }
.meta-card strong, .card strong { display: block; font-size: 1.3rem; margin-top: 6px; }
.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 18px 0 20px;
}
.tab {
  border: 1px solid var(--line);
  border-radius: 999px;
  background: rgba(255,255,255,0.78);
  color: var(--ink);
  padding: 10px 16px;
  cursor: pointer;
  font-weight: 700;
}
.tab.active {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}
.pane { display: none; }
.pane.active { display: grid; gap: 18px; }
.layout-two {
  display: grid;
  grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
  gap: 18px;
}
.panel, .result-block, .list-shell { padding: 18px; }
.panel h2, .result-block h2, .list-shell h2 {
  margin: 0 0 14px;
  font-size: 1.1rem;
}
label { display: grid; gap: 6px; font-size: 0.92rem; color: var(--muted); margin-bottom: 12px; }
input, select, textarea {
  width: 100%;
  border: 1px solid #c7d8df;
  border-radius: 14px;
  padding: 11px 12px;
  font: inherit;
  color: var(--ink);
  background: rgba(255,255,255,0.92);
}
textarea { min-height: 132px; resize: vertical; }
.inline-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
button.action {
  border: 0;
  border-radius: 14px;
  padding: 12px 16px;
  background: linear-gradient(135deg, var(--accent), #115e59);
  color: white;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}
button.secondary {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px 16px;
  background: rgba(255,255,255,0.8);
  color: var(--ink);
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}
.button-row { display: flex; flex-wrap: wrap; gap: 10px; }
.field-help {
  margin: -4px 0 10px;
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.55;
}
.preset-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin: 0 0 14px;
}
.preset-row .muted {
  font-size: 0.9rem;
}
button.preset-chip {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 12px;
  background: rgba(255,255,255,0.86);
  color: var(--ink);
  font: inherit;
  font-size: 0.92rem;
  font-weight: 700;
  cursor: pointer;
}
button.preset-chip.active {
  border-color: rgba(15,118,110,0.32);
  background: rgba(15,118,110,0.12);
  color: var(--accent);
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: 999px;
  font-weight: 700;
  background: rgba(15,118,110,0.12);
  color: var(--accent);
}
.status-pill.warn { background: rgba(201,107,31,0.14); color: #9a4f14; }
.status-pill.bad { background: rgba(185,28,28,0.12); color: #991b1b; }
.kv {
  display: grid;
  grid-template-columns: 180px 1fr;
  gap: 8px 12px;
  font-size: 0.95rem;
}
.kv div:nth-child(odd) { color: var(--muted); }
pre {
  margin: 0;
  padding: 14px;
  border-radius: 16px;
  background: #0d1b2a;
  color: #eef7ff;
  overflow-y: auto;
  overflow-x: hidden;
  max-width: 100%;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  line-height: 1.6;
  font-family: Consolas, "Cascadia Code", monospace;
  font-size: 0.9rem;
}
.response-surface, .preview-surface, .json-surface {
  min-height: 84px;
}
.selected-list, .summary-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 10px;
}
.selected-item {
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255,255,255,0.74);
  color: var(--ink);
}
.selected-item strong {
  display: block;
  margin-bottom: 4px;
}
.workspace-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}
.workspace-card {
  display: grid;
  gap: 12px;
}
.workspace-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.workspace-head h3 {
  margin: 0;
  font-size: 1.15rem;
}
.summary-row {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 14px;
  padding-top: 8px;
  border-top: 1px dashed rgba(16,39,58,0.12);
}
.summary-row:first-child {
  border-top: 0;
  padding-top: 0;
}
.summary-row strong {
  font-size: 0.94rem;
}
.summary-row span {
  color: var(--muted);
  text-align: right;
  overflow-wrap: anywhere;
}
.raw-toggle {
  margin-top: 6px;
}
.raw-toggle summary {
  cursor: pointer;
  color: var(--accent);
  font-weight: 700;
}
.raw-toggle pre {
  margin-top: 10px;
  max-height: 280px;
}
.status-inline {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.muted-callout {
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255,255,255,0.68);
  border: 1px dashed rgba(16,39,58,0.14);
  color: var(--muted);
}
.list-shell ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 10px;
  max-height: 560px;
  overflow-y: auto;
  overflow-x: hidden;
}
.audit-row {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px;
  cursor: pointer;
  background: rgba(255,255,255,0.72);
}
.audit-row:hover { border-color: var(--accent); }
.audit-row strong { display: block; }
.audit-row small { color: var(--muted); }
.muted { color: var(--muted); }
.warning-list { margin: 0; padding-left: 18px; color: #9a4f14; }
.guide-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.guide-card {
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.82);
}
.guide-card strong {
  display: block;
  margin-bottom: 6px;
}
.note {
  margin: 0 0 14px;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(15,118,110,0.18);
  background: rgba(15,118,110,0.06);
  color: var(--muted);
  line-height: 1.6;
}
.note code {
  font-family: Consolas, "Cascadia Code", monospace;
  color: var(--ink);
}
iframe.report-frame {
  width: 100%;
  min-height: 640px;
  border: 1px solid var(--line);
  border-radius: 20px;
  background: white;
}
@media (max-width: 980px) {
  .layout-two { grid-template-columns: 1fr; }
  .inline-grid { grid-template-columns: 1fr; }
  .kv { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<main>
  <header>
    <section class='hero'>
      <div class='eyebrow'>Mosaic Control Plane</div>
      <h1>Query, Audit, And Inspect The Runtime In One Place.</h1>
      <p>This local UI sits directly on top of the canonical MOSAIC runtime. Use it to run queries, inspect audited traces, check workspace readiness, and review the latest report and performance evidence without hopping between commands.</p>
      <section class='guide-grid'>
        <article class='guide-card'>
          <strong>Primary Output</strong>
          The <code>Query</code> pane is the main runtime surface. The <code>Response</code> box is the human-facing answer, and <code>Selected Chunks</code> are the supporting snippets MOSAIC chose.
        </article>
        <article class='guide-card'>
          <strong>Evidence Output</strong>
          The <code>Audit</code> pane is the explanation trace. It shows what was considered, what was denied, and why the final context set was selected.
        </article>
        <article class='guide-card'>
          <strong>Durable Output</strong>
          For downstream use, keep the generated files in <code>.mosaic/</code>: <code>report.html</code>, <code>benchmark.json</code>, <code>perf.json</code>, and <code>audit.db</code>. CLI and MCP can also return JSON directly.
        </article>
      </section>
      <section class='meta-grid' id='hero-meta'></section>
    </section>
  </header>

  <nav class='tabs'>
    <button class='tab active' data-pane='query-pane'>Query</button>
    <button class='tab' data-pane='corpus-pane'>Corpus</button>
    <button class='tab' data-pane='audit-pane'>Audit</button>
    <button class='tab' data-pane='workspace-pane'>Workspace</button>
    <button class='tab' data-pane='report-pane'>Report</button>
  </nav>

  <section class='pane active' id='query-pane'>
    <section class='layout-two'>
      <section class='panel'>
        <h2>Run Query</h2>
        <p class='note'>This is the main user-facing workflow. Ask a question about the currently loaded corpus, then read the answer in <code>Query Result</code>. <code>Permission Labels</code> define the permission scope for this request; in the bundled demo they happen to be labels like <code>public</code>, <code>analyst</code>, and <code>compliance</code>. Use <code>Budget</code> to cap context size, and <code>Conversation Budget</code> only when you are testing <code>mosaic_full</code>.</p>
        <label>Query
          <textarea id='query-text' placeholder='What workflow did the memo freeze?'></textarea>
        </label>
        <label>Permission Labels
          <input id='query-roles' value='public' placeholder='public, analyst'>
        </label>
        <p class='field-help' id='query-role-help'>This field controls the permission scope for the query. Replace <code>public</code> with a label like <code>analyst</code> or <code>compliance</code> to test denied versus allowed access.</p>
        <div class='preset-row' id='query-label-presets'></div>
        <section class='inline-grid'>
          <label>Strategy
            <select id='query-strategy'>
              <option value='mosaic_full'>mosaic_full</option>
              <option value='mosaic_no_ledger'>mosaic_no_ledger</option>
              <option value='mmr'>mmr</option>
              <option value='topk'>topk</option>
            </select>
          </label>
          <label>Candidate K
            <input id='query-candidate-k' type='number' value='50' min='1'>
          </label>
          <label>Budget
            <input id='query-budget' type='number' value='4096' min='1'>
          </label>
          <label>Conversation Budget
            <input id='query-conversation-budget' type='number' value='2200' min='1'>
          </label>
          <label>Lambda
            <input id='query-lam' type='number' step='0.05' value='0.5'>
          </label>
          <label>Cross-Turn Lambda
            <input id='query-cross-lambda' type='number' step='0.05' value='0.75'>
          </label>
        </section>
        <div class='button-row'>
          <button class='action' id='run-query'>Run Query</button>
          <button class='secondary' id='use-example'>Load Example</button>
        </div>
      </section>
      <section class='result-block'>
        <h2>Query Result</h2>
        <p class='note'>This pane is the answer output. <code>Response</code> is what an end user would read. <code>Selected Chunks</code> are the supporting snippets, and <code>audit_id</code> links this answer to a deeper trace you can inspect or export.</p>
        <div id='query-result-empty' class='muted'>Run a query to see status, answer, selected chunks, timings, and the linked audit trace.</div>
        <div id='query-result' hidden>
          <div id='query-status'></div>
          <div class='kv' id='query-kv' style='margin: 14px 0;'></div>
          <h3>Response</h3>
          <pre id='query-response' class='response-surface'></pre>
          <h3>Selected Chunks</h3>
          <ul id='query-selected' class='selected-list'></ul>
        </div>
      </section>
    </section>
  </section>

  <section class='pane' id='corpus-pane'>
    <section class='layout-two'>
      <section class='list-shell'>
        <h2>Loaded Documents</h2>
        <p class='note'>This is the source corpus MOSAIC is searching right now. Check this pane first if a query result looks surprising, because the loaded documents determine what the tool can retrieve.</p>
        <label>Filter Documents
          <input id='document-filter' placeholder='Search by title, id, permission label, or type'>
        </label>
        <div class='button-row' style='margin-bottom: 12px;'>
          <button class='action' id='refresh-documents'>Refresh Documents</button>
        </div>
        <ul id='document-list'></ul>
      </section>
      <section class='result-block'>
        <h2>Document Detail</h2>
        <p class='note'>Use this detail view to confirm document title, permission labels, source path, and sample text before you interpret a query result or permission gap.</p>
        <div id='document-empty' class='muted'>Choose a document to inspect its title, permission labels, counts, and preview text.</div>
        <div id='document-detail' hidden>
          <div class='kv' id='document-kv' style='margin: 14px 0;'></div>
          <h3>Preview</h3>
          <pre id='document-preview' class='preview-surface'></pre>
        </div>
      </section>
    </section>
  </section>

  <section class='pane' id='audit-pane'>
    <section class='layout-two'>
      <section class='list-shell'>
        <h2>Recent Audit Events</h2>
        <p class='note'>This is the explanation and governance side of the tool. Every live query should produce an <code>audit_id</code>; use that to inspect the exact runtime path MOSAIC took.</p>
        <div class='button-row' style='margin-bottom: 12px;'>
          <button class='action' id='refresh-audit'>Refresh</button>
        </div>
        <ul id='audit-list'></ul>
      </section>
      <section class='result-block'>
        <h2>Audit Trace</h2>
        <p class='note'>This pane is the evidence output, not the end-user answer. It explains the request outcome, selected context, denied candidates, and non-selection factors in machine-readable form.</p>
        <div id='audit-empty' class='muted'>Choose an audit event to inspect its compact trace summary.</div>
        <pre id='audit-trace' class='json-surface' hidden></pre>
      </section>
    </section>
  </section>

  <section class='pane' id='workspace-pane'>
    <section class='panel'>
      <p class='note'>This pane is for readiness, not for answers. Use it to confirm that the index, audit DB, benchmark, perf artifact, report, and live-eval backends are present before you debug a runtime issue.</p>
      <div class='button-row' style='margin-bottom: 14px;'>
        <button class='action' id='refresh-workspace'>Refresh Workspace</button>
      </div>
      <div id='workspace-summary' class='workspace-grid'></div>
      <div id='workspace-warnings' style='margin-top: 16px;'></div>
    </section>
  </section>

  <section class='pane' id='report-pane'>
    <section class='panel'>
      <p class='note'>This pane is the offline benchmark output. Use it to review system-wide results across many queries. For one live request, stay in <code>Query</code> and <code>Audit</code>.</p>
      <div class='button-row' style='margin-bottom: 14px;'>
        <button class='secondary' id='reload-report'>Reload Report</button>
      </div>
      <iframe class='report-frame' id='report-frame' src='/report'></iframe>
    </section>
  </section>
</main>
<script>
const state = { workspace: null, lastAuditId: null, documents: [], selectedDocumentId: null };

function el(id) { return document.getElementById(id); }
function parseRoles(raw) { return raw.split(',').map(v => v.trim()).filter(Boolean); }
function pretty(obj) { return JSON.stringify(obj, null, 2); }
function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
function formatText(value) {
  return escapeHtml(value).replace(/\\n/g, '<br>');
}
function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}
function countRowsText(rows, key) {
  if (!Array.isArray(rows) || !rows.length) return 'none';
  return rows.map(row => `${row[key] || 'unknown'}=${row.count || 0}`).join(', ');
}
function summaryRow(label, value) {
  return `<li class='summary-row'><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></li>`;
}
function rawToggle(value) {
  return `<details class='raw-toggle'><summary>Show raw details</summary><pre class='json-surface'>${escapeHtml(pretty(value))}</pre></details>`;
}
function workspaceCard(title, headline, tone, rows, rawValue) {
  return `
    <section class='card workspace-card'>
      <div class='workspace-head'>
        <div>
          <div class='eyebrow'>${escapeHtml(title)}</div>
          <h3>${escapeHtml(headline)}</h3>
        </div>
        <div class='status-inline'>${pill(tone === 'ok' ? 'ready' : tone === 'warn' ? 'partial' : 'missing', tone)}</div>
      </div>
      <ul class='summary-list'>${rows.join('')}</ul>
      ${rawToggle(rawValue)}
    </section>
  `;
}

function reportClientIssue(message) {
  console.error(message);
  const target = el('workspace-warnings');
  if (!target) {
    return;
  }
  const banner = `<div class='muted-callout'>${escapeHtml(message)}</div>`;
  if (!target.innerHTML.includes(message)) {
    target.innerHTML = banner + target.innerHTML;
  }
}

function safeBind(id, eventName, handler) {
  const target = el(id);
  if (!target) {
    reportClientIssue(`UI setup warning: missing element ${id}.`);
    return;
  }
  target.addEventListener(eventName, handler);
}

async function runSafely(taskName, task) {
  try {
    await task();
  } catch (error) {
    reportClientIssue(`UI warning: ${taskName} failed: ${error?.message || error}`);
  }
}

function corpusLabels(rows) {
  const labels = new Set();
  (rows || []).forEach(row => {
    (row.roles || []).forEach(label => {
      const value = String(label || '').trim();
      if (value) {
        labels.add(value);
      }
    });
  });
  return Array.from(labels).sort((left, right) => {
    if (left === 'public') return -1;
    if (right === 'public') return 1;
    return left.localeCompare(right);
  });
}

function currentPermissionLabels() {
  return parseRoles(el('query-roles').value);
}

function setPermissionLabels(labels) {
  el('query-roles').value = labels.join(', ');
  renderPermissionHelp();
  renderPermissionPresets();
}

function renderPermissionHelp() {
  const labels = currentPermissionLabels();
  const target = el('query-role-help');
  const rendered = labels.length ? labels.join(', ') : 'none';
  target.innerHTML = `Current permission scope: <code>${escapeHtml(rendered)}</code>. Change this field or click one of the labels below to test allowed versus denied access.`;
}

function renderPermissionPresets() {
  const target = el('query-label-presets');
  const labels = corpusLabels(state.documents);
  const active = new Set(currentPermissionLabels());
  if (!labels.length) {
    target.innerHTML = `<span class='muted'>Load a corpus to see suggested permission labels here.</span>`;
    return;
  }
  target.innerHTML = `<span class='muted'>Click a label to replace the current permission scope:</span>` + labels.map(label => `<button type='button' class='preset-chip${active.has(label) ? ' active' : ''}' data-role-label='${escapeHtml(label)}'>${escapeHtml(label)}</button>`).join('');
  target.querySelectorAll('.preset-chip').forEach(node => {
    node.addEventListener('click', () => setPermissionLabels([node.dataset.roleLabel]));
  });
}

function setActivePane(paneId) {
  document.querySelectorAll('.tab').forEach(button => {
    button.classList.toggle('active', button.dataset.pane === paneId);
  });
  document.querySelectorAll('.pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === paneId);
  });
}

function pill(text, kind='ok') {
  const klass = kind === 'bad' ? 'status-pill bad' : (kind === 'warn' ? 'status-pill warn' : 'status-pill');
  return `<span class='${klass}'>${text}</span>`;
}

function heroMeta(summary) {
  const checks = summary?.checks || {};
  const index = checks.index || {};
  const audit = checks.audit_db || {};
  const live = checks.live_eval || {};
  el('hero-meta').innerHTML = [
    `<div class='meta-card'><div class='eyebrow'>Index</div><strong>${index.documents || 0}</strong><span class='muted'>documents</span></div>`,
    `<div class='meta-card'><div class='eyebrow'>Chunks</div><strong>${index.chunks || 0}</strong><span class='muted'>retrieval units</span></div>`,
    `<div class='meta-card'><div class='eyebrow'>Audit</div><strong>${audit.request_count || 0}</strong><span class='muted'>requests logged</span></div>`,
    `<div class='meta-card'><div class='eyebrow'>Live Eval</div><strong>${live.live_answer_ready && live.live_judge_ready ? 'Ready' : 'Offline'}</strong><span class='muted'>backend readiness</span></div>`
  ].join('');
}

function workspaceCards(summary) {
  const checks = summary?.checks || {};
  const cards = [];

  const index = checks.index || {};
  cards.push(workspaceCard(
    'Index',
    index.exists ? `${index.documents || 0} documents` : 'Index missing',
    index.exists ? 'ok' : 'bad',
    [
      summaryRow('Chunks', index.chunks || 0),
      summaryRow('Vector Store', index.vector_store || 'n/a'),
      summaryRow('Corpus Source', index.corpus_source || 'n/a'),
      summaryRow('Embedding Model', index.embedding_model || 'n/a'),
    ],
    index,
  ));

  const audit = checks.audit_db || {};
  cards.push(workspaceCard(
    'Audit DB',
    audit.exists ? `${audit.request_count || 0} requests logged` : 'Audit DB missing',
    audit.exists ? 'ok' : 'warn',
    [
      summaryRow('Conversations', audit.conversation_count || 0),
      summaryRow('Latest Event', audit.latest_created_at || 'n/a'),
      summaryRow('Outcome Mix', countRowsText(audit.status_counts || [], 'status')),
    ],
    audit,
  ));

  const principals = checks.principal_map || {};
  cards.push(workspaceCard(
    'Principal Map',
    principals.exists ? `${principals.principal_count || 0} principals` : 'Principal map missing',
    principals.exists ? 'ok' : 'warn',
    [
      summaryRow('Bearer Tokens', principals.token_count || 0),
      summaryRow('HTTP Principals', Array.isArray(principals.http_principals) && principals.http_principals.length ? principals.http_principals.join(', ') : 'none'),
    ],
    principals,
  ));

  const report = checks.report || {};
  cards.push(workspaceCard(
    'Report',
    report.exists ? 'Report ready' : 'Report missing',
    report.exists ? 'ok' : 'warn',
    [
      summaryRow('Path', report.path || 'n/a'),
      summaryRow('Size', report.size_bytes ? formatBytes(report.size_bytes) : 'n/a'),
    ],
    report,
  ));

  const live = checks.live_eval || {};
  const liveReady = Boolean(live.live_answer_ready && live.live_judge_ready);
  cards.push(workspaceCard(
    'Live Eval',
    liveReady ? 'Anthropic ready' : 'Offline / proxy mode',
    liveReady ? 'ok' : 'warn',
    [
      summaryRow('Answer Ready', live.live_answer_ready ? 'yes' : `no (${live.live_answer_unavailable_reason || 'unavailable'})`),
      summaryRow('Judge Ready', live.live_judge_ready ? 'yes' : `no (${live.live_judge_unavailable_reason || 'unavailable'})`),
      summaryRow('Answer Model', live.answer_model || 'n/a'),
      summaryRow('Judge Model', live.judge_model || 'n/a'),
    ],
    live,
  ));

  if (checks.benchmark) {
    const benchmark = checks.benchmark;
    cards.push(workspaceCard(
      'Benchmark',
      benchmark.exists ? `${benchmark.strategy_count || 0} strategies` : 'Benchmark missing',
      benchmark.exists ? 'ok' : 'warn',
      [
        summaryRow('Single-Turn Rows', benchmark.single_turn_rows || 0),
        summaryRow('Failure Rows', benchmark.failure_rows || 0),
        summaryRow('Multi-Turn Rows', benchmark.multi_turn_rows || 0),
      ],
      benchmark,
    ));
  }

  if (checks.perf) {
    const perf = checks.perf;
    cards.push(workspaceCard(
      'Perf',
      perf.exists ? `${Number(perf.throughput_qps || 0).toFixed(2)} qps` : 'Perf missing',
      perf.exists ? 'ok' : 'warn',
      [
        summaryRow('Requests', perf.request_count || 0),
        summaryRow('Total p95', perf.latency_ms?.total?.p95 ? `${perf.latency_ms.total.p95} ms` : 'n/a'),
        summaryRow('Outcome Mix', countRowsText(perf.status_counts || [], 'status')),
      ],
      perf,
    ));
  }

  el('workspace-summary').innerHTML = cards.join('');
  const warnings = summary?.warnings || [];
  el('workspace-warnings').innerHTML = warnings.length
    ? `<h2>Warnings</h2><ul class='warning-list'>${warnings.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`
    : `<div class='muted-callout'>No workspace warnings.</div>`;
}

async function refreshWorkspace() {
  const response = await fetch('/api/workspace');
  const payload = await response.json();
  state.workspace = payload;
  heroMeta(payload);
  workspaceCards(payload);
}

function renderQueryResponse(payload) {
  el('query-result-empty').hidden = true;
  el('query-result').hidden = false;
  const kind = payload.status === 'success' ? 'ok' : (payload.status === 'permission_gap' ? 'warn' : 'bad');
  el('query-status').innerHTML = `${pill(payload.status, kind)} ${pill(payload.classification, kind)}`;
  const kvRows = [
    ['Audit ID', payload.audit_id],
    ['Conversation ID', payload.conversation_id || 'n/a'],
    ['Tokens Used', payload.tokens_used],
    ['Remaining Budget', payload.remaining_budget],
  ];
  if (payload.required_role) {
    kvRows.push([payload.status === 'permission_gap' ? 'Required Role' : 'Higher-Tier Material', payload.required_role]);
  }
  el('query-kv').innerHTML = kvRows.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join('');
  el('query-response').innerHTML = formatText(payload.response || payload.answer || '');
  const selectedTitles = Array.isArray(payload.selected_chunk_titles) ? payload.selected_chunk_titles : [];
  el('query-selected').innerHTML = selectedTitles.length
    ? selectedTitles.map((title, index) => `<li class='selected-item'><strong>Snippet ${index + 1}</strong><span>${escapeHtml(title)}</span></li>`).join('')
    : `<li class='muted-callout'>No chunks were selected for this request.</li>`;
  state.lastAuditId = payload.audit_id;
  refreshAudit();
}

async function runQuery() {
  const strategy = el('query-strategy').value;
  const payload = {
    query: el('query-text').value.trim(),
    requested_roles: parseRoles(el('query-roles').value),
    strategy,
    candidate_k: Number(el('query-candidate-k').value),
    token_budget: strategy === 'mosaic_full' ? Number(el('query-conversation-budget').value) : Number(el('query-budget').value),
    lam: Number(el('query-lam').value),
    cross_turn_lambda: Number(el('query-cross-lambda').value)
  };
  const response = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) {
    el('query-result-empty').hidden = true;
    el('query-result').hidden = false;
    el('query-status').innerHTML = pill('request_failed', 'bad');
    el('query-kv').innerHTML = '';
    el('query-response').innerHTML = formatText(result.error || 'Request failed.');
    el('query-selected').innerHTML = `<li class='muted-callout'>No chunks were selected.</li>`;
    return;
  }
  renderQueryResponse(result);
  await refreshWorkspace();
}

function renderDocumentDetail(row) {
  el('document-empty').hidden = true;
  el('document-detail').hidden = false;
  el('document-kv').innerHTML = [
    ['Title', row.title],
    ['Document ID', row.document_id],
    ['Permission Labels', (row.roles || []).join(', ') || 'n/a'],
    ['Chunks', row.chunk_count],
    ['Tokens', row.token_count],
    ['Type', row.doc_type || 'n/a'],
    ['Source', row.source_path || 'n/a']
  ].map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join('');
  el('document-preview').textContent = row.preview || '(no preview available)';
}

function documentMatches(row, filterText) {
  if (!filterText) {
    return true;
  }
  const haystack = [
    row.title,
    row.document_id,
    (row.roles || []).join(' '),
    row.doc_type || '',
    row.source_path || ''
  ].join(' ').toLowerCase();
  return haystack.includes(filterText);
}

function documentRow(row) {
  return `
    <li>
      <article class='audit-row' data-document-id='${row.document_id}'>
        <strong>${row.title}</strong>
        <small>${row.document_id}</small>
        <div>${row.chunk_count} chunk(s) | ${(row.roles || []).join(', ') || 'n/a'}</div>
      </article>
    </li>
  `;
}

function bindDocumentRows(rows) {
  const target = el('document-list');
  if (!Array.isArray(rows) || rows.length === 0) {
    target.innerHTML = `<li class='muted'>No documents matched the current filter.</li>`;
    return;
  }
  target.innerHTML = rows.map(documentRow).join('');
  target.querySelectorAll('.audit-row').forEach(node => {
    node.addEventListener('click', () => loadDocument(node.dataset.documentId));
  });
}

function applyDocumentFilter() {
  const filterText = el('document-filter').value.trim().toLowerCase();
  const filtered = state.documents.filter(row => documentMatches(row, filterText));
  bindDocumentRows(filtered);
  if (state.selectedDocumentId) {
    const selected = state.documents.find(row => row.document_id === state.selectedDocumentId);
    if (selected) {
      renderDocumentDetail(selected);
    }
  }
}

async function refreshDocuments() {
  const response = await fetch('/api/documents');
  const rows = await response.json();
  state.documents = Array.isArray(rows) ? rows : [];
  renderPermissionPresets();
  applyDocumentFilter();
  if (!state.selectedDocumentId && state.documents.length) {
    loadDocument(state.documents[0].document_id);
  }
}

function loadDocument(documentId) {
  const row = state.documents.find(item => item.document_id === documentId);
  if (!row) {
    return;
  }
  state.selectedDocumentId = documentId;
  renderDocumentDetail(row);
}

function auditRow(row) {
  return `
    <li>
      <article class='audit-row' data-audit-id='${row.audit_id}'>
        <strong>${row.status} / ${row.classification}</strong>
        <small>${row.created_at || ''}</small>
        <div>${row.query}</div>
      </article>
    </li>
  `;
}

async function refreshAudit() {
  const response = await fetch('/api/audit/events?limit=20');
  const rows = await response.json();
  const target = el('audit-list');
  if (!Array.isArray(rows) || rows.length === 0) {
    target.innerHTML = `<li class='muted'>No audit events available.</li>`;
    return;
  }
  target.innerHTML = rows.map(auditRow).join('');
  target.querySelectorAll('.audit-row').forEach(node => {
    node.addEventListener('click', () => loadAudit(node.dataset.auditId));
  });
  if (state.lastAuditId) {
    loadAudit(state.lastAuditId);
  }
}

async function loadAudit(auditId) {
  const response = await fetch(`/api/audit/trace?audit_id=${encodeURIComponent(auditId)}&summary_only=1&candidate_limit=10`);
  const payload = await response.json();
  el('audit-empty').hidden = true;
  el('audit-trace').hidden = false;
  el('audit-trace').textContent = pretty(payload);
}

function loadExample() {
  el('query-text').value = 'What workflow did the memo freeze?';
  setPermissionLabels(['analyst']);
  el('query-strategy').value = 'mosaic_no_ledger';
  el('query-budget').value = '700';
}

function resizeReportFrame() {
  const frame = el('report-frame');
  if (!frame) {
    return;
  }
  try {
    const doc = frame.contentDocument || frame.contentWindow?.document;
    if (!doc) {
      return;
    }
    const height = Math.max(
      doc.documentElement?.scrollHeight || 0,
      doc.body?.scrollHeight || 0,
      640,
    );
    frame.style.height = `${height + 16}px`;
  } catch (error) {
    reportClientIssue(`UI warning: report resize failed: ${error?.message || error}`);
  }
}

function reloadReport() {
  const frame = el('report-frame');
  frame.src = `/report?ts=${Date.now()}`;
}


function bootUi() {
  document.querySelectorAll('.tab').forEach(button => {
    button.addEventListener('click', () => setActivePane(button.dataset.pane));
  });
  safeBind('run-query', 'click', () => { void runSafely('query execution', runQuery); });
  safeBind('refresh-documents', 'click', () => { void runSafely('document refresh', refreshDocuments); });
  safeBind('document-filter', 'input', applyDocumentFilter);
  safeBind('query-roles', 'input', () => { renderPermissionHelp(); renderPermissionPresets(); });
  safeBind('refresh-audit', 'click', () => { void runSafely('audit refresh', refreshAudit); });
  safeBind('refresh-workspace', 'click', () => { void runSafely('workspace refresh', refreshWorkspace); });
  safeBind('reload-report', 'click', reloadReport);
  safeBind('report-frame', 'load', resizeReportFrame);
  safeBind('use-example', 'click', loadExample);
  loadExample();
  renderPermissionHelp();
  renderPermissionPresets();
  void runSafely('workspace refresh', refreshWorkspace);
  void runSafely('document refresh', refreshDocuments);
  void runSafely('audit refresh', refreshAudit);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootUi);
} else {
  bootUi();
}
</script>
</body>
</html>
"""


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({'error': message}, status_code=status_code)


def _coerce_roles(value: Any) -> list[str]:
    if value is None:
        return ['public']
    if isinstance(value, str):
        return [part.strip() for part in value.split(',') if part.strip()] or ['public']
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()] or ['public']
    return ['public']


def _preview_text(text: str, limit: int = 280) -> str:
    compact = ' '.join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + '...'


def _document_rows(chunks: list[Chunk]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        row = rows.setdefault(
            chunk.document_id,
            {
                'document_id': chunk.document_id,
                'title': chunk.metadata.get('title', chunk.document_id),
                'roles': set(),
                'chunk_count': 0,
                'token_count': 0,
                'doc_type': chunk.metadata.get('doc_type'),
                'source_path': chunk.metadata.get('source_path') or chunk.metadata.get('path'),
                'preview': '',
                'sample_chunk_ids': [],
            },
        )
        row['roles'].update(chunk.roles)
        row['chunk_count'] += 1
        row['token_count'] += chunk.token_count
        if not row['preview']:
            row['preview'] = _preview_text(chunk.text)
        if row['title'] == chunk.document_id and chunk.metadata.get('title'):
            row['title'] = chunk.metadata['title']
        if not row['doc_type'] and chunk.metadata.get('doc_type'):
            row['doc_type'] = chunk.metadata['doc_type']
        if not row['source_path'] and (chunk.metadata.get('source_path') or chunk.metadata.get('path')):
            row['source_path'] = chunk.metadata.get('source_path') or chunk.metadata.get('path')
        if len(row['sample_chunk_ids']) < 5:
            row['sample_chunk_ids'].append(chunk.id)

    payload = []
    for row in rows.values():
        payload.append(
            {
                'document_id': row['document_id'],
                'title': row['title'],
                'roles': sorted(row['roles']),
                'chunk_count': row['chunk_count'],
                'token_count': row['token_count'],
                'doc_type': row['doc_type'],
                'source_path': row['source_path'],
                'preview': row['preview'],
                'sample_chunk_ids': row['sample_chunk_ids'],
            }
        )
    payload.sort(key=lambda item: (str(item['title']).lower(), item['document_id']))
    return payload


def build_ui_app(
    *,
    index_path: str | Path,
    audit_db_path: str | Path | None = None,
    principal_map_path: str | Path | None = None,
    calibration_suite_path: str | Path | None = None,
    report_path: str | Path = '.mosaic/report.html',
    benchmark_path: str | Path | None = None,
    perf_path: str | Path | None = None,
    answer_model: str,
    judge_model: str,
) -> Starlette:
    resolved_index_path = Path(index_path)
    resolved_report_path = Path(report_path)
    resolved_benchmark_path = Path(benchmark_path) if benchmark_path is not None else None
    resolved_perf_path = Path(perf_path) if perf_path is not None else None
    resolved_principal_map = Path(principal_map_path) if principal_map_path is not None else Path('examples/principals.sample.json')
    resolver = PrincipalResolver.from_file(resolved_principal_map if resolved_principal_map.exists() else None)
    service = MosaicService(
        resolved_index_path,
        audit_store=SQLiteAuditStore(audit_db_path) if audit_db_path else None,
        ledger_store=SQLiteLedgerStore(audit_db_path) if audit_db_path else None,
        principal_resolver=resolver,
        calibration_suite_path=calibration_suite_path,
        report_path=resolved_report_path,
    )
    document_rows = _document_rows(service.index.chunks)

    async def home(_: Request) -> Response:
        return HTMLResponse(UI_HTML, headers={'Cache-Control': 'no-store'})

    async def healthz(_: Request) -> Response:
        return PlainTextResponse('ok')

    async def report(request: Request) -> Response:
        if not resolved_report_path.exists():
            return HTMLResponse('<html><body><p>No report has been generated yet.</p></body></html>', headers={'Cache-Control': 'no-store'})
        return HTMLResponse(resolved_report_path.read_text(encoding='utf-8-sig'), headers={'Cache-Control': 'no-store'})

    async def workspace(_: Request) -> Response:
        payload = build_workspace_summary(
            index_path=resolved_index_path,
            audit_db_path=audit_db_path or '.mosaic/audit.db',
            principal_map_path=resolved_principal_map,
            report_path=resolved_report_path,
            answer_model=answer_model,
            judge_model=judge_model,
            benchmark_path=resolved_benchmark_path,
            perf_path=resolved_perf_path,
        )
        return JSONResponse(payload)

    async def query(request: Request) -> Response:
        payload = await request.json()
        try:
            response = service.query(
                QueryRequest(
                    query=str(payload.get('query', '')),
                    principal_id=str(payload.get('principal_id', 'local-ui')),
                    requested_roles=_coerce_roles(payload.get('requested_roles') or payload.get('roles')),
                    strategy=str(payload.get('strategy', 'mosaic_full')),
                    candidate_k=int(payload.get('candidate_k', 50)),
                    token_budget=int(payload.get('token_budget', 4096)),
                    conversation_id=payload.get('conversation_id'),
                    turn=int(payload['turn']) if payload.get('turn') is not None else None,
                    lam=float(payload.get('lam', 0.5)),
                    cross_turn_lambda=float(payload.get('cross_turn_lambda', 0.75)),
                    return_trace=bool(payload.get('return_trace', False)),
                ),
                transport='ui',
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return _json_error(str(exc), status_code=400)
        return JSONResponse(response.to_dict())

    async def documents(_: Request) -> Response:
        return JSONResponse(document_rows)

    async def audit_events(request: Request) -> Response:
        if service.audit_store is None:
            return JSONResponse([])
        params = request.query_params
        rows = service.list_audit_events(
            limit=int(params.get('limit', '20')),
            status=params.get('status'),
            classification=params.get('classification'),
            principal_id=params.get('principal_id'),
            conversation_id=params.get('conversation_id'),
        )
        return JSONResponse(rows)

    async def audit_trace(request: Request) -> Response:
        if service.audit_store is None:
            return _json_error('Audit store is not configured.', status_code=400)
        params = request.query_params
        audit_id = params.get('audit_id')
        if not audit_id:
            return _json_error('audit_id is required.', status_code=400)
        try:
            payload = service.get_audit_trace(
                audit_id,
                include_text=params.get('include_text', '0') == '1',
                candidate_limit=int(params['candidate_limit']) if params.get('candidate_limit') else None,
                summary_only=params.get('summary_only', '0') == '1',
            )
        except RuntimeError as exc:
            return _json_error(str(exc), status_code=400)
        return JSONResponse(payload)

    routes = [
        Route('/', home),
        Route('/healthz', healthz),
        Route('/report', report),
        Route('/api/workspace', workspace),
        Route('/api/query', query, methods=['POST']),
        Route('/api/documents', documents),
        Route('/api/audit/events', audit_events),
        Route('/api/audit/trace', audit_trace),
    ]
    return Starlette(debug=False, routes=routes)


def serve_ui(
    *,
    index_path: str | Path,
    audit_db_path: str | Path | None = None,
    principal_map_path: str | Path | None = None,
    calibration_suite_path: str | Path | None = None,
    report_path: str | Path = '.mosaic/report.html',
    benchmark_path: str | Path | None = None,
    perf_path: str | Path | None = None,
    answer_model: str,
    judge_model: str,
    host: str = '127.0.0.1',
    port: int = 8090,
) -> None:
    app = build_ui_app(
        index_path=index_path,
        audit_db_path=audit_db_path,
        principal_map_path=principal_map_path,
        calibration_suite_path=calibration_suite_path,
        report_path=report_path,
        benchmark_path=benchmark_path,
        perf_path=perf_path,
        answer_model=answer_model,
        judge_model=judge_model,
    )
    uvicorn.run(app, host=host, port=port, log_level='warning')
