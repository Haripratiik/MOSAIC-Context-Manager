from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from plotly import graph_objects as go
from plotly.io import to_html
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .audit import load_governance_source
from .evaluator import aggregate_results, aggregate_multiturn
from .utils import load_json


def _load_rows(input_paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in input_paths:
        rows.extend(load_json(path))
    return rows


def _cards_html(summary_rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in summary_rows:
        cards.append(
            """
            <section class='card'>
              <div class='eyebrow'>{strategy}</div>
              <h3>{faithfulness:.3f} FAI</h3>
              <p><strong>TTFT</strong> {ttft_ms:.1f} ms</p>
              <p><strong>Tokens</strong> {tokens_used:.1f}</p>
              <p><strong>Redundancy</strong> {redundancy_score:.3f}</p>
              <p><strong>Efficiency</strong> {efficiency:.6f}</p>
            </section>
            """.format(**row)
        )
    return "\n".join(cards)


def _benchmark_cards_html(summary_rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in summary_rows:
        cards.append(
            """
            <section class='card'>
              <div class='eyebrow'>{strategy}</div>
              <h3>{faithfulness:.3f} FAI</h3>
              <p><strong>TTFT</strong> {ttft_ms:.1f} ms</p>
              <p><strong>Redundancy</strong> {redundancy_score:.3f}</p>
              <p><strong>Efficiency</strong> {efficiency:.6f}</p>
              <p><strong>BGU</strong> {budget_utilization:.3f}</p>
              <p><strong>XTR</strong> {cross_turn_redundancy:.3f}</p>
              <p><strong>CFA</strong> {cumulative_faithfulness:.3f}</p>
            </section>
            """.format(
                strategy=row.get("strategy", "unknown"),
                faithfulness=row.get("faithfulness", 0.0),
                ttft_ms=row.get("ttft_ms", 0.0),
                redundancy_score=row.get("redundancy_score", 0.0),
                efficiency=row.get("efficiency", 0.0),
                budget_utilization=row.get("budget_utilization", 0.0),
                cross_turn_redundancy=row.get("cross_turn_redundancy", 0.0),
                cumulative_faithfulness=row.get("cumulative_faithfulness", 0.0),
            )
        )
    return "\n".join(cards)


def _build_pareto_figure(pareto_rows: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    if pareto_rows:
        figure.add_trace(
            go.Scatter(
                x=[row["ttft_ms"] for row in pareto_rows],
                y=[row["faithfulness"] for row in pareto_rows],
                mode="lines+markers+text",
                text=[f"lambda={row['lam']:.2f}" for row in pareto_rows],
                textposition="top center",
                line={"color": "#0f766e", "width": 3},
                marker={"size": 9, "color": "#0f766e"},
                name="MOSAIC Pareto",
            )
        )
    figure.update_layout(title="Quality vs TTFT Pareto Frontier", xaxis_title="Average TTFT (ms)", yaxis_title="Average Faithfulness", template="plotly_white", height=420)
    return figure


def _build_category_figure(category_rows: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    categories = sorted({row["category"] for row in category_rows})
    strategies = sorted({row["strategy"] for row in category_rows})
    for strategy in strategies:
        values = []
        for category in categories:
            match = next((row for row in category_rows if row["strategy"] == strategy and row["category"] == category), None)
            values.append(match["faithfulness"] if match else 0.0)
        figure.add_trace(go.Bar(name=strategy, x=categories, y=values))
    figure.update_layout(title="Per-Category Faithfulness", barmode="group", template="plotly_white", height=420)
    return figure


def _select_heatmap_query(rows: list[dict[str, Any]]) -> str | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("category") == "redundancy_trap":
            grouped[row["query_id"]].append(row)
    if not grouped:
        return None
    return max(grouped, key=lambda query_id: sum(item["redundancy_score"] for item in grouped[query_id]))


def _short_title(value: Any, limit: int = 72) -> str:
    compact = ' '.join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + '...'


def _heatmap_titles(row: dict[str, Any]) -> list[str]:
    titles = row.get("selected_chunk_titles") or []
    if titles:
        return [str(title) for title in titles]
    chunk_ids = row.get("selected_chunk_ids") or []
    return [str(chunk_id).split("::")[0] for chunk_id in chunk_ids] or ["No selected chunks"]


def _build_heatmap_artifacts(rows: list[dict[str, Any]]) -> tuple[go.Figure, str]:
    chosen_query = _select_heatmap_query(rows)
    strategies = sorted({row["strategy"] for row in rows})
    cols = 2 if len(strategies) > 1 else 1
    subplot_rows = max(1, (len(strategies) + cols - 1) // cols)
    figure = make_subplots(
        rows=subplot_rows,
        cols=cols,
        subplot_titles=strategies,
        horizontal_spacing=0.08 if cols > 1 else 0.05,
        vertical_spacing=0.18,
    )
    if chosen_query is None:
        figure.update_layout(title="Redundancy Heatmap", template="plotly_white", height=360)
        return figure, "<p class='chart-note'>No redundancy-trap rows were available, so there is no representative heatmap yet.</p>"

    legend_cards: list[str] = []
    query_label = str(chosen_query).replace('_', ' ')
    for index, strategy in enumerate(strategies, start=1):
        subplot_row = (index - 1) // cols + 1
        subplot_col = (index - 1) % cols + 1
        row = next((item for item in rows if item["query_id"] == chosen_query and item["strategy"] == strategy), None)
        matrix = row.get("similarity_matrix") if row else None
        titles = _heatmap_titles(row or {})
        if not matrix:
            z = [[0.0]]
            short_labels = ["C1"]
            full_titles = ["No selected chunks"]
            hover_text = [["No redundancy matrix available."]]
        else:
            width = min(len(matrix), len(matrix[0]) if matrix else 0, len(titles))
            if width <= 0:
                z = [[0.0]]
                short_labels = ["C1"]
                full_titles = ["No selected chunks"]
                hover_text = [["No redundancy matrix available."]]
            else:
                z = [[float(value) for value in line[:width]] for line in matrix[:width]]
                full_titles = titles[:width]
                short_labels = [f"C{item + 1}" for item in range(width)]
                hover_text = [
                    [
                        f"{full_titles[y_index]}<br>vs<br>{full_titles[x_index]}<br>Similarity {z[y_index][x_index]:.2f}"
                        for x_index in range(width)
                    ]
                    for y_index in range(width)
                ]
        figure.add_trace(
            go.Heatmap(
                z=z,
                x=short_labels,
                y=short_labels,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                colorscale=[[0.0, "#effcf7"], [0.35, "#99f6e4"], [0.7, "#14b8a6"], [1.0, "#134e4a"]],
                showscale=index == len(strategies),
                zmin=0.0,
                zmax=1.0,
                xgap=3,
                ygap=3,
            ),
            row=subplot_row,
            col=subplot_col,
        )
        legend_items = ''.join(
            f"<li><strong>C{item + 1}</strong><span>{escape(_short_title(title, 108))}</span></li>"
            for item, title in enumerate(full_titles)
        )
        legend_cards.append(
            f"""
            <section class='legend-card'>
              <div class='eyebrow'>{escape(strategy)}</div>
              <h4>Chunk Legend</h4>
              <ul>{legend_items or '<li><strong>C1</strong><span>No selected chunks.</span></li>'}</ul>
            </section>
            """
        )

    figure.update_layout(
        title=f"Redundancy Heatmap for {query_label}",
        template="plotly_white",
        height=420 * subplot_rows + 80,
        margin={"t": 96, "r": 48, "b": 36, "l": 36},
    )
    figure.update_xaxes(title_text="Selected snippets", tickangle=0, automargin=True)
    figure.update_yaxes(title_text="Selected snippets", automargin=True)
    legend_html = f"""
  <p class='chart-note'>Representative query: <strong>{escape(query_label)}</strong>. The matrix labels are shortened to <strong>C1</strong>, <strong>C2</strong>, and so on so the chart stays readable. Hover the heatmap for full pairwise titles, then use the legend below to map each code back to a snippet.</p>
  <section class='legend-grid'>
    {''.join(legend_cards)}
  </section>
"""
    return figure, legend_html


def _build_heatmap_figure(rows: list[dict[str, Any]]) -> go.Figure:
    return _build_heatmap_artifacts(rows)[0]


def _build_query_table(rows: list[dict[str, Any]]) -> go.Figure:
    ordered = sorted(rows, key=lambda item: (item["query_id"], item["strategy"]))
    figure = go.Figure(
        data=[
            go.Table(
                header={"values": ["Query", "Strategy", "Category", "FAI", "RED", "TTFT ms", "Tokens"], "fill_color": "#cdece5", "align": "left"},
                cells={
                    "values": [
                        [row["query_id"] for row in ordered],
                        [row["strategy"] for row in ordered],
                        [row["category"] for row in ordered],
                        [row["faithfulness"] for row in ordered],
                        [row["redundancy_score"] for row in ordered],
                        [row["ttft_ms"] for row in ordered],
                        [row["tokens_used"] for row in ordered],
                    ],
                    "fill_color": "#ffffff",
                    "align": "left",
                },
            )
        ]
    )
    header_height = 34
    row_height = 30
    table_height = max(220, 96 + header_height + row_height * max(len(ordered), 1))
    figure.update_layout(title="Per-Query Results", height=table_height, margin={"t": 56, "r": 18, "b": 18, "l": 18})
    return figure


def _build_failure_accuracy_figure(summary: dict[str, Any]) -> go.Figure:
    rows = summary.get("per_type", [])
    figure = go.Figure([go.Bar(x=[row["label"] for row in rows], y=[row["accuracy"] for row in rows], marker_color="#0f766e")])
    figure.update_layout(title="Classification Accuracy by Failure Type", yaxis_title="Accuracy", template="plotly_white", height=360)
    return figure


def _build_gap_scatter_figure(rows: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    labels = sorted({row["ground_truth_type"] for row in rows})
    palette = {"TRUE_UNKNOWN": "#94a3b8", "PERMISSION_GAP": "#dc2626", "RETRIEVAL_FAILURE": "#0f766e"}
    for label in labels:
        scoped = [row["scoped_score"] for row in rows if row["ground_truth_type"] == label]
        opened = [row["open_score"] for row in rows if row["ground_truth_type"] == label]
        text = [row["query_id"] for row in rows if row["ground_truth_type"] == label]
        figure.add_trace(go.Scatter(x=scoped, y=opened, mode="markers", name=label, text=text, marker={"size": 10, "color": palette.get(label, "#334155")}))
    figure.update_layout(title="Open vs Scoped Retrieval Scores", xaxis_title="Scoped score", yaxis_title="Open score", template="plotly_white", height=420)
    return figure


def _build_confusion_figure(summary: dict[str, Any]) -> go.Figure:
    labels = ["TRUE_UNKNOWN", "PERMISSION_GAP", "RETRIEVAL_FAILURE"]
    matrix = summary.get("confusion_matrix", {})
    z = [[matrix.get(actual, {}).get(predicted, 0) for predicted in labels] for actual in labels]
    figure = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale="Teal", zmin=0))
    figure.update_layout(title="Failure-Type Confusion Matrix", template="plotly_white", height=380)
    return figure


def _failure_examples_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    for label in ["TRUE_UNKNOWN", "PERMISSION_GAP", "RETRIEVAL_FAILURE"]:
        row = next((item for item in rows if item["ground_truth_type"] == label and item["correct"]), None)
        if row is None:
            row = next((item for item in rows if item["ground_truth_type"] == label), None)
        if row is None:
            continue
        cards.append(
            f"""
            <section class='example'>
              <div class='eyebrow'>{label}</div>
              <p><strong>Query</strong> {row['query']}</p>
              <p><strong>Before MOSAIC</strong> I don't know.</p>
              <p><strong>After MOSAIC</strong> {row['response']}</p>
            </section>
            """
        )
    return "\n".join(cards)


def _build_multiturn_curve_figure(rows: list[dict[str, Any]], metric: str, title: str, yaxis: str) -> go.Figure:
    figure = go.Figure()
    strategies = sorted({row["strategy"] for row in rows})
    for strategy in strategies:
        series = [row for row in rows if row["strategy"] == strategy]
        series.sort(key=lambda item: item["turn"])
        figure.add_trace(go.Scatter(x=[row["turn"] for row in series], y=[row[metric] for row in series], mode="lines+markers", name=strategy))
    figure.update_layout(title=title, xaxis_title="Turn", yaxis_title=yaxis, template="plotly_white", height=380)
    return figure


def _representative_scenario_html(rows: list[dict[str, Any]]) -> str:
    scenario_ids = sorted({row["scenario_id"] for row in rows if row.get("scenario_id")})
    if not scenario_ids:
        return ""
    scenario_id = scenario_ids[0]
    topk_rows = sorted([row for row in rows if row["scenario_id"] == scenario_id and row["strategy"] == "topk"], key=lambda item: item["turn"])
    mosaic_rows = sorted([row for row in rows if row["scenario_id"] == scenario_id and row["strategy"] == "mosaic_full"], key=lambda item: item["turn"])
    turns = sorted({row["turn"] for row in topk_rows + mosaic_rows})
    html = ["<table class='scenario-table'><thead><tr><th>Turn</th><th>Top-k Answer</th><th>MOSAIC Full Answer</th></tr></thead><tbody>"]
    for turn in turns:
        topk = next((row for row in topk_rows if row["turn"] == turn), None)
        mosaic = next((row for row in mosaic_rows if row["turn"] == turn), None)
        html.append(f"<tr><td>{turn}</td><td>{(topk or {}).get('answer', '')}</td><td>{(mosaic or {}).get('answer', '')}</td></tr>")
    html.append("</tbody></table>")
    return "".join(html)


def _build_latency_breakdown(summary_rows: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    strategies = [row["strategy"] for row in summary_rows]
    figure.add_trace(go.Bar(name="retrieval", x=strategies, y=[row.get("retrieval_ms", 0.0) for row in summary_rows]))
    figure.add_trace(go.Bar(name="optimization", x=strategies, y=[row.get("optimization_ms", 0.0) for row in summary_rows]))
    figure.add_trace(go.Bar(name="ttft", x=strategies, y=[row.get("ttft_ms", 0.0) for row in summary_rows]))
    figure.update_layout(title="Latency Breakdown", barmode="stack", template="plotly_white", height=380)
    return figure


def _build_overview_table(summary_rows: list[dict[str, Any]]) -> go.Figure:
    ordered = sorted(summary_rows, key=lambda item: item["strategy"])
    figure = go.Figure(
        data=[
            go.Table(
                header={"values": ["Strategy", "FAI", "RED", "TTFT", "EFF", "XTR", "BGU", "CFA"], "fill_color": "#cdece5", "align": "left"},
                cells={
                    "values": [
                        [row["strategy"] for row in ordered],
                        [row.get("faithfulness", 0.0) for row in ordered],
                        [row.get("redundancy_score", 0.0) for row in ordered],
                        [row.get("ttft_ms", 0.0) for row in ordered],
                        [row.get("efficiency", 0.0) for row in ordered],
                        [row.get("cross_turn_redundancy", 0.0) for row in ordered],
                        [row.get("budget_utilization", 0.0) for row in ordered],
                        [row.get("cumulative_faithfulness", 0.0) for row in ordered],
                    ],
                    "fill_color": "#ffffff",
                    "align": "left",
                },
            )
        ]
    )
    figure.update_layout(title="System Overview", height=320)
    return figure


def _build_governance_outcome_figure(summary: dict[str, Any]) -> go.Figure:
    rows = summary.get("outcomes", [])
    figure = go.Figure([go.Bar(x=[row["status"] for row in rows], y=[row["count"] for row in rows], marker_color="#0f766e")])
    figure.update_layout(title="Request Volume by Outcome", yaxis_title="Requests", template="plotly_white", height=340)
    return figure


def _build_deny_matrix_figure(summary: dict[str, Any]) -> go.Figure:
    rows = summary.get("deny_matrix", [])
    callers = sorted({row["caller_roles"] for row in rows})
    required_roles = sorted({row["required_role"] for row in rows})
    if not callers or not required_roles:
        figure = go.Figure()
        figure.update_layout(title="Permission-Gap Deny Matrix", template="plotly_white", height=340)
        return figure
    z = []
    for caller in callers:
        z.append([next((row["count"] for row in rows if row["caller_roles"] == caller and row["required_role"] == role), 0) for role in required_roles])
    figure = go.Figure(go.Heatmap(z=z, x=required_roles, y=callers, colorscale="Teal", zmin=0))
    figure.update_layout(title="Permission-Gap Deny Matrix", xaxis_title="Required role", yaxis_title="Caller roles", template="plotly_white", height=360)
    return figure


def _governance_examples_html(summary: dict[str, Any]) -> str:
    examples = []
    sample_success = summary.get("sample_success")
    sample_gap = summary.get("sample_permission_gap")
    if sample_success:
        examples.append(
            f"""
            <section class='example'>
              <div class='eyebrow'>Success Trace</div>
              <p><strong>Query</strong> {sample_success['request']['query']}</p>
              <p><strong>Outcome</strong> {sample_success['outcome']['response']}</p>
              <p><strong>Selected</strong> {', '.join(sample_success['outcome'].get('selected_chunk_titles', []))}</p>
            </section>
            """
        )
    if sample_gap:
        examples.append(
            f"""
            <section class='example'>
              <div class='eyebrow'>Permission Gap Trace</div>
              <p><strong>Query</strong> {sample_gap['request']['query']}</p>
              <p><strong>Outcome</strong> {sample_gap['outcome']['response']}</p>
              <p><strong>Required Role</strong> {sample_gap['outcome'].get('required_role', 'unknown')}</p>
            </section>
            """
        )
    return ''.join(examples)


def _governance_section_html(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ''
    outcome_html = to_html(_build_governance_outcome_figure(summary), include_plotlyjs=False, full_html=False, config={"responsive": True})
    deny_html = to_html(_build_deny_matrix_figure(summary), include_plotlyjs=False, full_html=False, config={"responsive": True})
    examples_html = _governance_examples_html(summary)
    return f"""
  <h2 class='section-title'>Governance</h2>
  <section class='cards'>
    <section class='card'><div class='eyebrow'>Audit</div><h3>{summary.get('request_count', 0)} Requests</h3><p><strong>Denied candidates / request</strong> {summary.get('avg_denied_candidates', 0.0):.3f}</p></section>
  </section>
  <section class='panel'>{outcome_html}</section>
  <section class='panel'>{deny_html}</section>
  <section class='example-grid'>{examples_html}</section>
"""


def _count_rows_html(rows: list[dict[str, Any]], key: str) -> str:
    if not rows:
        return 'none'
    return ', '.join(f"{row.get(key, 'unknown')}={row.get('count', 0)}" for row in rows)


def _evaluation_provenance_section_html(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ''
    ttft_counts = _count_rows_html(summary.get('ttft_backend_counts', []), 'backend')
    faith_counts = _count_rows_html(summary.get('faithfulness_backend_counts', []), 'backend')
    ttft_reasons = _count_rows_html(summary.get('ttft_proxy_reason_counts', []), 'reason')
    faith_reasons = _count_rows_html(summary.get('faithfulness_proxy_reason_counts', []), 'reason')
    return f"""
  <h2 class='section-title'>Measurement Provenance</h2>
  <section class='cards'>
    <section class='card'><div class='eyebrow'>Backends</div><h3>{'Live' if summary.get('all_live') else 'Mixed / Proxy'}</h3><p><strong>TTFT</strong> {ttft_counts}</p><p><strong>Faithfulness</strong> {faith_counts}</p></section>
    <section class='card'><div class='eyebrow'>Readiness</div><h3>{'Ready' if summary.get('live_answer_ready') and summary.get('live_judge_ready') else 'Offline / Partial'}</h3><p><strong>Answer</strong> {summary.get('answer_model')}</p><p><strong>Judge</strong> {summary.get('judge_model')}</p></section>
  </section>
  <section class='panel'><p><strong>TTFT fallback reasons</strong> {ttft_reasons}</p><p><strong>Faithfulness fallback reasons</strong> {faith_reasons}</p></section>
"""


def _perf_section_html(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ''
    metadata = payload.get('metadata', {})
    summary = payload.get('summary', {})
    latency = summary.get('latency_ms', {})
    return f"""
  <h2 class='section-title'>Performance Evidence</h2>
  <section class='cards'>
    <section class='card'><div class='eyebrow'>Throughput</div><h3>{summary.get('throughput_qps', 0.0):.3f} qps</h3><p><strong>Requests</strong> {summary.get('request_count', 0)}</p><p><strong>Concurrency</strong> {metadata.get('concurrency', 1)}</p></section>
    <section class='card'><div class='eyebrow'>Latency</div><h3>{latency.get('total', {}).get('p95', 0.0):.1f} ms p95</h3><p><strong>Retrieval p95</strong> {latency.get('retrieval', {}).get('p95', 0.0):.1f}</p><p><strong>Optimization p95</strong> {latency.get('optimization', {}).get('p95', 0.0):.1f}</p></section>
    <section class='card'><div class='eyebrow'>Statuses</div><h3>{metadata.get('strategy', 'unknown')}</h3><p><strong>Outcome mix</strong> {_count_rows_html(summary.get('status_counts', []), 'status')}</p><p><strong>Categories</strong> {', '.join(metadata.get('categories', []))}</p></section>
  </section>
"""


def _report_styles() -> str:
    return """
:root { --bg-top: #f8fbff; --bg-bottom: #edf7f1; --ink: #12324a; --muted: #4f6b7a; --card: rgba(255,255,255,0.92); --line: #d7e5ea; --accent: #0f766e; }
* { box-sizing: border-box; }
body { margin: 0; font-family: Bahnschrift, "Segoe UI Variable", "Segoe UI", sans-serif; color: var(--ink); line-height: 1.6; background: radial-gradient(circle at top left, #ffffff 0%, var(--bg-top) 35%, var(--bg-bottom) 100%); }
main { max-width: 1440px; margin: 0 auto; padding: 40px 28px 72px; }
header { display: grid; gap: 12px; margin-bottom: 28px; }
header h1 { margin: 0; font-family: Georgia, 'Times New Roman', serif; font-size: clamp(2.4rem, 4vw, 4.2rem); letter-spacing: -0.03em; }
header p { margin: 0; color: var(--muted); font-size: 1.03rem; line-height: 1.7; max-width: 86ch; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 24px 0 36px; }
.card, .panel, .example, .legend-card { background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 18px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06); }
.panel { margin-top: 20px; overflow: hidden; }
.card h3, .legend-card h4 { margin: 8px 0 10px; }
.card p, .example p, .panel p, .scenario-table td, .scenario-table th, .legend-card span { overflow-wrap: anywhere; }
.card p:last-child, .example p:last-child { margin-bottom: 0; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.72rem; color: var(--accent); }
.section-title { font-size: 1.4rem; margin: 10px 0 0; }
.example-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-top: 16px; }
.scenario-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.scenario-table th, .scenario-table td { border: 1px solid var(--line); padding: 10px; text-align: left; vertical-align: top; }
.chart-note { margin: 12px 0 0; padding: 12px 14px; border-radius: 16px; background: rgba(15,118,110,0.08); border: 1px dashed rgba(15,118,110,0.22); color: var(--muted); }
.legend-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 14px; }
.legend-card ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }
.legend-card li { display: grid; grid-template-columns: 52px minmax(0, 1fr); gap: 12px; align-items: start; padding-top: 10px; border-top: 1px dashed var(--line); }
.legend-card li:first-child { padding-top: 0; border-top: 0; }
.legend-card strong { display: inline-flex; align-items: center; justify-content: center; min-height: 32px; padding: 0 10px; border-radius: 999px; background: rgba(15,118,110,0.12); color: var(--accent); }
.js-plotly-plot, .plotly-graph-div { width: 100% !important; }
"""


def _render_comprehensive_report(payload: dict[str, Any], output_path: str | Path, title: str) -> Path:
    single_rows = payload.get("single_turn_rows", [])
    failure_rows = payload.get("failure_rows", [])
    multi_turn_rows = payload.get("multi_turn_rows", [])
    pareto_rows = payload.get("pareto_rows", [])
    summary = payload.get("summary", {})
    strategy_rows = summary.get("strategy_metrics", [])
    category_rows = summary.get("category_metrics", [])
    failure_summary = summary.get("classification", {})
    multi_turn_summary = aggregate_multiturn(multi_turn_rows)
    governance_summary = payload.get("_governance")
    provenance_summary = payload.get("metadata", {}).get("evaluation_provenance")
    perf_summary = payload.get("_perf")

    summary_html = _benchmark_cards_html(strategy_rows)
    pareto_html = to_html(_build_pareto_figure(pareto_rows), include_plotlyjs=False, full_html=False, config={"responsive": True})
    category_html = to_html(_build_category_figure(category_rows), include_plotlyjs=False, full_html=False, config={"responsive": True})
    heatmap_figure, heatmap_legend_html = _build_heatmap_artifacts(single_rows)
    heatmap_html = to_html(heatmap_figure, include_plotlyjs=False, full_html=False, config={"responsive": True})
    query_html = to_html(_build_query_table(single_rows), include_plotlyjs=False, full_html=False, config={"responsive": True})
    accuracy_html = to_html(_build_failure_accuracy_figure(failure_summary), include_plotlyjs=False, full_html=False, config={"responsive": True})
    scatter_html = to_html(_build_gap_scatter_figure(failure_rows), include_plotlyjs=False, full_html=False, config={"responsive": True})
    confusion_html = to_html(_build_confusion_figure(failure_summary), include_plotlyjs=False, full_html=False, config={"responsive": True})
    failure_examples = _failure_examples_html(failure_rows)
    budget_html = to_html(_build_multiturn_curve_figure(multi_turn_summary.get("per_turn", []), "remaining_budget", "Budget Remaining by Turn", "Remaining budget"), include_plotlyjs=False, full_html=False, config={"responsive": True})
    cfa_html = to_html(_build_multiturn_curve_figure(multi_turn_summary.get("per_turn", []), "faithfulness", "CFA Across Turns", "Faithfulness"), include_plotlyjs=False, full_html=False, config={"responsive": True})
    xtr_html = to_html(_build_multiturn_curve_figure(multi_turn_summary.get("per_turn", []), "cross_turn_redundancy", "Cross-Turn Redundancy by Turn", "XTR"), include_plotlyjs=False, full_html=False, config={"responsive": True})
    latency_html = to_html(_build_latency_breakdown(strategy_rows), include_plotlyjs=False, full_html=False, config={"responsive": True})
    overview_html = to_html(_build_overview_table(strategy_rows), include_plotlyjs=False, full_html=False, config={"responsive": True})

    best_strategy = max(strategy_rows, key=lambda item: item.get("faithfulness", 0.0)) if strategy_rows else None
    narrative = ""
    if best_strategy is not None:
        narrative = f"The best current baseline is <strong>{best_strategy['strategy']}</strong> with FAI {best_strategy.get('faithfulness', 0.0):.3f}, TTFT {best_strategy.get('ttft_ms', 0.0):.1f} ms, and BGU {best_strategy.get('budget_utilization', 0.0):.3f}."

    html = f"""
<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{title}</title>
<style>{_report_styles()}</style>
<script>{get_plotlyjs()}</script>
</head>
<body>
<main>
  <header>
    <div class='eyebrow'>Minimum Overlap Semantic Assembly with Information Coverage</div>
    <h1>{title}</h1>
    <p>Comprehensive offline dashboard covering the assembler, gap classifier, and conversation ledger over the synthetic financial-services benchmark.</p>
    <p>{narrative}</p>
  </header>
  {_evaluation_provenance_section_html(provenance_summary)}
  {_perf_section_html(perf_summary)}
  <h2 class='section-title'>Optimizer Performance</h2>
  <section class='cards'>{summary_html}</section>
  <section class='panel'>{pareto_html}</section>
  <section class='panel'>{category_html}</section>
  <section class='panel'>{heatmap_html}{heatmap_legend_html}</section>
  <section class='panel'>{query_html}</section>
  <h2 class='section-title'>Failure Classification</h2>
  <section class='panel'>{accuracy_html}</section>
  <section class='panel'>{scatter_html}</section>
  <section class='panel'>{confusion_html}</section>
  <section class='example-grid'>{failure_examples}</section>
  <h2 class='section-title'>Multi-Turn Coherence</h2>
  <section class='panel'>{budget_html}</section>
  <section class='panel'>{cfa_html}</section>
  <section class='panel'>{xtr_html}</section>
  <section class='panel'>{_representative_scenario_html(multi_turn_rows)}</section>
  <h2 class='section-title'>System Overview</h2>
  <section class='panel'>{latency_html}</section>
  <section class='panel'>{overview_html}</section>
  {_governance_section_html(governance_summary)}
</main>
</body>
</html>
"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def render_report(
    input_paths: list[str | Path],
    output_path: str | Path,
    pareto_path: str | Path | None = None,
    title: str = "MOSAIC Report",
    benchmark_path: str | Path | None = None,
    audit_db_path: str | Path | None = None,
    audit_export_path: str | Path | None = None,
    perf_path: str | Path | None = None,
) -> Path:
    if benchmark_path is not None:
        payload = load_json(benchmark_path)
        payload["_governance"] = load_governance_source(audit_db_path=audit_db_path, audit_export_path=audit_export_path)
        payload["_perf"] = load_json(perf_path) if perf_path else None
        return _render_comprehensive_report(payload, output_path=output_path, title=title)

    rows = _load_rows(input_paths)
    summary = aggregate_results(rows)
    pareto_rows = load_json(pareto_path) if pareto_path else []
    perf_payload = load_json(perf_path) if perf_path else None
    summary_html = _cards_html(summary["strategies"])
    provenance_html = _evaluation_provenance_section_html(summary.get("backend_provenance"))
    perf_html = _perf_section_html(perf_payload)
    pareto_html = to_html(_build_pareto_figure(pareto_rows), include_plotlyjs=False, full_html=False, config={"responsive": True})
    category_html = to_html(_build_category_figure(summary["categories"]), include_plotlyjs=False, full_html=False, config={"responsive": True})
    heatmap_figure, heatmap_legend_html = _build_heatmap_artifacts(rows)
    heatmap_html = to_html(heatmap_figure, include_plotlyjs=False, full_html=False, config={"responsive": True})
    table_html = to_html(_build_query_table(rows), include_plotlyjs=False, full_html=False, config={"responsive": True})
    html = f"""
<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{title}</title>
<style>{_report_styles()}</style>
<script>{get_plotlyjs()}</script>
</head>
<body>
<main>
  <header>
    <div class='eyebrow'>Minimum Overlap Semantic Assembly with Information Coverage</div>
    <h1>{title}</h1>
    <p>Compact offline report for MOSAIC strategy comparisons, provenance, and runtime evidence.</p>
  </header>
  {provenance_html}
  {perf_html}
  <section class='cards'>{summary_html}</section>
  <section class='panel'>{pareto_html}</section>
  <section class='panel'>{category_html}</section>
  <section class='panel'>{heatmap_html}{heatmap_legend_html}</section>
  <section class='panel'>{table_html}</section>
</main>
</body>
</html>
"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target
