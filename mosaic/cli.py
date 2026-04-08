from __future__ import annotations

"""CLI entrypoints for building, querying, auditing, benchmarking, and serving MOSAIC."""

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
import threading
from textwrap import dedent
from time import perf_counter
import webbrowser

from .audit import JsonLedgerStore, SQLiteAuditStore, SQLiteLedgerStore
from .auth import AuthError, BearerTokenAuthProvider, PrincipalResolver
from .corpus_builder import generate_corpus, generate_eval_suite
from .evaluator import (
    DEFAULT_ANTHROPIC_ANSWER_MODEL,
    DEFAULT_ANTHROPIC_JUDGE_MODEL,
    aggregate_results,
    run_benchmark,
    run_eval,
    run_pareto,
    select_strategy,
)
from .ingestor import SUPPORTED_SOURCE_EXTENSIONS, ingest_directory, load_index
from .mcp_server import serve_mcp
from .perf import run_perf
from .report import render_report
from .ui import serve_ui
from .retriever import hybrid_retrieve
from .service import MosaicService, QueryRequest
from .utils import DEFAULT_EMBEDDING_MODEL, dump_json, load_json, parse_roles
from .workspace import build_workspace_summary

DEFAULT_HTTP_ALLOWED_ORIGINS = ["http://127.0.0.1", "http://localhost"]


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _parse_lam_values(raw: str | None, max_lam: float, steps: int) -> list[float]:
    if raw:
        return [float(part.strip()) for part in raw.split(",") if part.strip()]
    if steps <= 1:
        return [round(max_lam, 4)]
    return [round((max_lam / (steps - 1)) * index, 4) for index in range(steps)]


def _parse_extensions(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return SUPPORTED_SOURCE_EXTENSIONS
    resolved = []
    for part in raw.split(','):
        cleaned = part.strip().lower()
        if not cleaned:
            continue
        resolved.append(cleaned if cleaned.startswith('.') else f'.{cleaned}')
    if not resolved:
        raise ValueError('At least one source extension is required.')
    return tuple(resolved)


def _count_rows_to_text(rows: list[dict[str, object]], key: str) -> str:
    return ', '.join(f"{row[key]}={row['count']}" for row in rows) if rows else 'none'


def _demo_step(step: int, total: int, message: str) -> None:
    print(f"[demo {step}/{total}] {message}", flush=True)


def _demo_options(args: argparse.Namespace) -> dict[str, object]:
    if not args.fast:
        return {
            "embedding_model": args.embedding_model,
            "vector_store": args.vector_store,
            "steps": args.steps,
            "perf_limit": args.perf_limit,
            "perf_iterations": args.perf_iterations,
            "perf_warmup": args.perf_warmup,
            "perf_concurrency": args.perf_concurrency,
        }
    return {
        "embedding_model": "hash",
        "vector_store": "json",
        "steps": min(args.steps, 3),
        "perf_limit": min(args.perf_limit, 2),
        "perf_iterations": 1,
        "perf_warmup": 0,
        "perf_concurrency": 1,
    }


def _schedule_url_open(url: str, *, delay_s: float = 1.0) -> None:
    timer = threading.Timer(delay_s, lambda: _open_url(url))
    timer.daemon = True
    timer.start()


def _open_path(path: Path) -> bool:
    target = path.resolve()
    try:
        return bool(webbrowser.open(target.as_uri()))
    except Exception:
        return False


def _open_url(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write_json_output(path: str | Path | None, payload: object, *, label: str | None = None) -> None:
    if not path:
        return
    dump_json(path, payload)
    if label:
        print(f"{label} written to {path}")


def _resolver(args: argparse.Namespace) -> PrincipalResolver:
    return PrincipalResolver.from_file(getattr(args, "principal_map", None))


def _audit_store(args: argparse.Namespace) -> SQLiteAuditStore | None:
    audit_db = getattr(args, "audit_db", None)
    return SQLiteAuditStore(audit_db) if audit_db else None


def _ledger_store(args: argparse.Namespace):
    if getattr(args, "ledger_state", None):
        return JsonLedgerStore(args.ledger_state)
    if getattr(args, "audit_db", None):
        return SQLiteLedgerStore(args.audit_db)
    return None


def _service(
    args: argparse.Namespace,
    index_path: str | Path | None = None,
    *,
    principal_resolver: PrincipalResolver | None = None,
) -> MosaicService:
    return MosaicService(
        index_path or args.index,
        audit_store=_audit_store(args),
        ledger_store=_ledger_store(args),
        principal_resolver=principal_resolver or _resolver(args),
        calibration_suite_path=getattr(args, "calibration_suite", None),
        report_path=getattr(args, "report_path", None),
    )


def _query_budget(args: argparse.Namespace) -> int:
    return args.conversation_budget if args.strategy == "mosaic_full" else args.budget


def _normalized_mcp_path(raw: str) -> str:
    return raw if raw.startswith("/") else f"/{raw}"


def _ui_calibration_suite_path(explicit: str | None, index_path: str | Path) -> str | None:
    if explicit:
        return explicit
    default_suite = Path('corpus/eval_suite.json')
    try:
        if Path(index_path).resolve() == Path('.mosaic/index.json').resolve() and default_suite.exists():
            return str(default_suite)
    except FileNotFoundError:
        pass
    return None


def _format_status_line(response) -> list[str]:
    lines = [
        f"status: {response.status}",
        f"classification: {response.classification}",
    ]
    if response.required_role:
        lines.append(f"required_role: {response.required_role}")
    if response.hints:
        lines.append(f"hints: {'; '.join(response.hints)}")
    return lines


def _format_selected_chunk_lines(response) -> list[str]:
    counts = Counter(response.selected_chunk_titles)
    lines: list[str] = []
    for chunk_id, title in zip(response.selected_chunk_ids, response.selected_chunk_titles):
        if counts[title] > 1:
            lines.append(f"- {title} [{chunk_id}]")
        else:
            lines.append(f"- {title}")
    return lines


def _add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help_text: str,
    description: str,
) -> argparse.ArgumentParser:
    return subparsers.add_parser(
        name,
        help=help_text,
        description=description,
        formatter_class=HelpFormatter,
    )


def _add_index_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--index",
        required=required,
        help="Path to the retrieval index JSON file produced by `mosaic ingest`.",
    )


def _add_roles_argument(parser: argparse.ArgumentParser, *, default: str = "public") -> None:
    parser.add_argument(
        "--roles",
        default=default,
        help="Comma-separated roles for local CLI use. On HTTP MCP, requested roles are intersected with the authenticated principal.",
    )


def _add_principal_map_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--principal-map",
        help="Path to a principals JSON file. See examples/principals.sample.json for the expected shape.",
    )


def _add_audit_db_argument(parser: argparse.ArgumentParser, *, default: str = ".mosaic/audit.db") -> None:
    parser.add_argument(
        "--audit-db",
        default=default,
        help="SQLite audit database. For `mosaic_full`, this also acts as the default ledger store unless `--ledger-state` is supplied.",
    )


def cmd_generate_corpus(args: argparse.Namespace) -> int:
    catalog = generate_corpus(args.output_dir, catalog_path=args.catalog, clean=args.clean)
    print(f"Generated {len(catalog)} documents in {args.output_dir}")
    return 0


def cmd_generate_eval(args: argparse.Namespace) -> int:
    catalog = load_json(args.catalog)
    suite = generate_eval_suite(catalog, output_path=args.output)
    print(f"Generated {suite['metadata']['total_queries']} evaluation items in {args.output}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    if not args.source and not args.manifest:
        raise ValueError('Provide a source directory or --manifest for ingest.')
    index = ingest_directory(
        source_dir=args.source,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        vector_store=args.vector_store,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        manifest_path=args.manifest,
        source_extensions=_parse_extensions(args.extensions),
    )
    corpus_source = index.config.get('corpus_source', 'directory')
    document_count = index.config.get('document_count', len({chunk.document_id for chunk in index.chunks}))
    print(f"Ingested {len(index.chunks)} chunks across {document_count} documents into {args.output} ({corpus_source})")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    index = load_index(args.index)
    results = hybrid_retrieve(
        args.query,
        index.chunks,
        user_roles=parse_roles(args.roles),
        k=args.k,
        alpha=args.alpha,
        embedding_model=str(index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
    )
    for chunk in results:
        print(f"{chunk.id} score={chunk.relevance:.4f} roles={','.join(chunk.roles)}")
        print(chunk.text[:220])
        print()
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    if args.conversation_id and args.strategy != "mosaic_full":
        print("Note: --conversation-id is only used when --strategy mosaic_full is selected.", file=sys.stderr)
    service = _service(args)
    response = service.query(
        QueryRequest(
            query=args.query,
            principal_id=args.principal_id,
            requested_roles=parse_roles(args.roles),
            strategy=args.strategy,
            candidate_k=args.candidate_k,
            token_budget=_query_budget(args),
            conversation_id=args.conversation_id,
            turn=args.turn,
            lam=args.lam,
            cross_turn_lambda=args.cross_turn_lambda,
            return_trace=args.include_trace or bool(args.output),
        ),
        transport="cli",
    )
    payload = response.to_dict()
    _write_json_output(args.output, payload, label="Query payload")

    for line in _format_status_line(response):
        print(line)
    print()
    print(response.response)

    if response.selected_chunk_titles:
        print()
        print("selected_chunks:")
        for line in _format_selected_chunk_lines(response):
            print(line)

    print()
    print(f"audit_id: {response.audit_id}")
    if response.conversation_id:
        print(f"conversation_id: {response.conversation_id}")
    if response.remaining_budget is not None:
        print(f"remaining_budget: {response.remaining_budget}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    index = load_index(args.index)
    user_roles = parse_roles(args.roles)
    candidates = hybrid_retrieve(
        args.query,
        index.chunks,
        user_roles=user_roles,
        k=args.candidate_k,
        alpha=args.alpha,
        embedding_model=str(index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
    )
    result = select_strategy(
        args.strategy,
        candidates,
        user_roles=user_roles,
        token_budget=args.budget,
        lam=args.lam,
        use_jax=None if args.use_jax == "auto" else args.use_jax == "true",
        ledger=None,
        cross_turn_lambda=args.cross_turn_lambda,
    )
    payload = [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "title": chunk.metadata.get("title", chunk.document_id),
            "roles": chunk.roles,
            "relevance": round(chunk.relevance, 4),
            "token_count": chunk.token_count,
        }
        for chunk in result.selected
    ]
    _write_json_output(args.output, payload, label="Optimization payload")
    print(f"Selected {len(result.selected)} chunks via {result.backend}")
    for row in payload:
        print(row)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    results = run_eval(
        index_path=args.index,
        eval_suite_path=args.suite,
        strategy=args.strategy,
        token_budget=args.budget,
        candidate_k=args.candidate_k,
        lam=args.lam,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        use_jax=None if args.use_jax == "auto" else args.use_jax == "true",
        output_path=args.output,
        require_live_answer=args.require_live_answer,
        require_live_judge=args.require_live_judge,
    )
    summary = aggregate_results(results)
    provenance = summary.get('backend_provenance', {})
    print(f"Wrote {len(results)} eval rows to {args.output}")
    print(f"TTFT backends: {_count_rows_to_text(provenance.get('ttft_backend_counts', []), 'backend')}")
    print(f"Faithfulness backends: {_count_rows_to_text(provenance.get('faithfulness_backend_counts', []), 'backend')}")
    return 0


def cmd_pareto(args: argparse.Namespace) -> int:
    lam_values = _parse_lam_values(args.lam_values, args.max_lam, args.steps)
    frontier = run_pareto(
        index_path=args.index,
        eval_suite_path=args.suite,
        token_budget=args.budget,
        candidate_k=args.candidate_k,
        lam_values=lam_values,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        use_jax=None if args.use_jax == "auto" else args.use_jax == "true",
        output_path=args.output,
        require_live_answer=args.require_live_answer,
        require_live_judge=args.require_live_judge,
    )
    print(f"Wrote {len(frontier)} Pareto rows to {args.output}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    payload = run_benchmark(
        index_path=args.index,
        eval_suite_path=args.suite,
        token_budget=args.budget,
        conversation_budget=args.conversation_budget,
        candidate_k=args.candidate_k,
        lam=args.lam,
        cross_turn_lambda=args.cross_turn_lambda,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        use_jax=None if args.use_jax == "auto" else args.use_jax == "true",
        output_path=args.output,
        pareto_steps=args.steps,
        max_lam=args.max_lam,
        require_live_answer=args.require_live_answer,
        require_live_judge=args.require_live_judge,
    )
    provenance = payload.get('metadata', {}).get('evaluation_provenance', {})
    print(
        f"Wrote benchmark results to {args.output} with {len(payload.get('failure_rows', []))} failure queries "
        f"and {len(payload.get('multi_turn_rows', []))} multi-turn rows"
    )
    print(f"TTFT backends: {_count_rows_to_text(provenance.get('ttft_backend_counts', []), 'backend')}")
    print(f"Faithfulness backends: {_count_rows_to_text(provenance.get('faithfulness_backend_counts', []), 'backend')}")
    return 0


def cmd_perf(args: argparse.Namespace) -> int:
    if args.query and args.suite:
        raise ValueError('Use either --suite or one or more --query values for `mosaic perf`, not both.')
    if not args.query and not args.suite:
        raise ValueError('Provide --suite or at least one --query value for `mosaic perf`.')

    adhoc_queries = None
    if args.query:
        adhoc_queries = [
            {
                'id': f'adhoc-{index + 1}',
                'query': query,
                'roles': parse_roles(args.roles),
                'category': 'adhoc',
            }
            for index, query in enumerate(args.query)
        ]

    payload = run_perf(
        args.index,
        suite_path=None if adhoc_queries is not None else args.suite,
        queries=adhoc_queries,
        strategy=args.strategy,
        token_budget=args.budget,
        conversation_budget=args.conversation_budget,
        candidate_k=args.candidate_k,
        lam=args.lam,
        cross_turn_lambda=args.cross_turn_lambda,
        iterations=args.iterations,
        warmup=args.warmup,
        concurrency=args.concurrency,
        limit=args.limit,
        categories=args.category,
        include_multi_turn=args.include_multi_turn,
        audit_db_path=args.audit_db,
        principal_map_path=args.principal_map,
        calibration_suite_path=args.calibration_suite,
        output_path=args.output,
    )
    summary = payload['summary']
    print(f"Perf results written to {args.output}")
    print(f"throughput_qps: {summary['throughput_qps']}")
    print(f"total_p95_ms: {summary['latency_ms']['total']['p95']}")
    print(f"statuses: {_count_rows_to_text(summary.get('status_counts', []), 'status')}")
    return 0


def cmd_audit_list(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.list_audit_events(
        limit=args.limit,
        status=args.status,
        classification=args.classification,
        principal_id=args.principal_id,
        conversation_id=args.conversation_id,
    )
    if args.output:
        _write_json_output(args.output, rows, label="Audit list")
    else:
        _print_json(rows)
    return 0


def cmd_audit_show(args: argparse.Namespace) -> int:
    service = _service(args)
    payload = service.get_audit_trace(
        args.audit_id,
        include_text=args.include_text,
        candidate_limit=args.candidate_limit,
        summary_only=args.summary,
    )
    if args.output:
        _write_json_output(args.output, payload, label="Audit trace")
    else:
        _print_json(payload)
    return 0


def cmd_audit_export(args: argparse.Namespace) -> int:
    service = _service(args)
    payload = service.export_audit(
        audit_id=args.audit_id,
        conversation_id=args.conversation_id,
        limit=args.limit,
        include_text=args.include_text,
    )
    if args.output:
        _write_json_output(args.output, payload, label="Audit export")
    else:
        _print_json(payload)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    payload = build_workspace_summary(
        index_path=args.index,
        audit_db_path=args.audit_db,
        principal_map_path=args.principal_map,
        report_path=args.report_path,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        strict=args.strict,
        require_live_answer=args.require_live_answer,
        require_live_judge=args.require_live_judge,
        benchmark_path=getattr(args, 'benchmark_path', None),
        perf_path=getattr(args, 'perf_path', None),
    )
    _write_json_output(args.output, payload, label="Doctor report")
    if not args.output:
        _print_json(payload)
    return 0 if payload["ok"] else 1


def cmd_ui(args: argparse.Namespace) -> int:
    url = f"http://{args.host}:{args.port}"
    calibration_suite = _ui_calibration_suite_path(args.calibration_suite, args.index)
    print(f"Serving MOSAIC UI at {url}")
    print(f"index: {args.index}")
    if args.audit_db:
        print(f"audit_db: {args.audit_db}")
    if args.report_path:
        print(f"report: {args.report_path}")
    if calibration_suite:
        print(f"calibration_suite: {calibration_suite}")
    if args.open_ui:
        _schedule_url_open(url)
        print(f"Auto-open scheduled for {url}")
    serve_ui(
        index_path=args.index,
        audit_db_path=args.audit_db,
        principal_map_path=args.principal_map,
        calibration_suite_path=calibration_suite,
        report_path=args.report_path,
        benchmark_path=args.benchmark_path,
        perf_path=args.perf_path,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        host=args.host,
        port=args.port,
    )
    return 0

def cmd_serve_mcp(args: argparse.Namespace) -> int:
    resolver = _resolver(args)
    auth_required = not args.disable_http_auth
    normalized_path = _normalized_mcp_path(args.path)
    allowed_origins = args.allow_origin or list(DEFAULT_HTTP_ALLOWED_ORIGINS)

    if args.transport == "http" and auth_required and not resolver.has_bearer_tokens():
        print(
            "HTTP MCP auth is enabled, but no bearer tokens are configured. "
            "Provide --principal-map with token-bearing principals or use --disable-http-auth for local-only development.",
            file=sys.stderr,
        )
        return 2

    service = _service(args, principal_resolver=resolver)
    if args.transport == "http":
        print(f"Serving MOSAIC MCP over HTTP at http://{args.host}:{args.port}{normalized_path}")
        print(f"auth: {'bearer token required' if auth_required else 'disabled'}")
        print(f"allowed_origins: {', '.join(allowed_origins)}")
    else:
        print("Serving MOSAIC MCP over stdio. Connect your MCP client to this process.")

    serve_mcp(
        service,
        transport="streamable-http" if args.transport == "http" else "stdio",
        principal_resolver=resolver,
        auth_provider=None if args.disable_http_auth else BearerTokenAuthProvider(),
        host=args.host,
        port=args.port,
        path=normalized_path,
        allowed_origins=allowed_origins,
        require_auth=auth_required,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if not args.benchmark and not args.inputs:
        print("Provide --benchmark or at least one JSON results file via --inputs.", file=sys.stderr)
        return 2
    target = render_report(
        args.inputs or [],
        output_path=args.output,
        pareto_path=args.pareto,
        title=args.title,
        benchmark_path=args.benchmark,
        audit_db_path=args.audit_db,
        audit_export_path=args.audit_export,
        perf_path=args.perf,
    )
    print(f"Report written to {target}")
    if args.open_report:
        opened = _open_path(target)
        print(f"Auto-open {'succeeded' if opened else 'was skipped'} for {target}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    started = perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(args.corpus_dir)
    catalog_path = Path(args.catalog)
    eval_path = Path(args.eval_suite)
    index_path = out_dir / "index.json"
    benchmark_path = out_dir / "benchmark.json"
    pareto_path = out_dir / "pareto.json"
    perf_path = out_dir / "perf.json"
    report_target = out_dir / "report.html"
    audit_db_path = Path(args.audit_db) if args.audit_db else out_dir / "audit.db"
    options = _demo_options(args)

    if args.fast:
        print("[demo] Using fast preset: hash embeddings, JSON vector store, and lighter perf settings.", flush=True)
    if args.clean and not args.audit_db and audit_db_path.exists():
        audit_db_path.unlink()

    _demo_step(1, 6, f"Generating corpus in {corpus_dir}...")
    step_started = perf_counter()
    generate_corpus(corpus_dir, catalog_path=catalog_path, clean=args.clean)
    print(f"  corpus ready in {(perf_counter() - step_started):.1f}s", flush=True)

    _demo_step(2, 6, f"Generating evaluation suite at {eval_path}...")
    step_started = perf_counter()
    generate_eval_suite(load_json(catalog_path), output_path=eval_path)
    print(f"  eval suite ready in {(perf_counter() - step_started):.1f}s", flush=True)

    _demo_step(3, 6, f"Ingesting documents with embedding_model={options['embedding_model']} and vector_store={options['vector_store']}...")
    step_started = perf_counter()
    ingest_directory(
        source_dir=corpus_dir,
        output_path=index_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=str(options['embedding_model']),
        embedding_dimensions=args.embedding_dimensions,
        vector_store=str(options['vector_store']),
        chroma_dir=args.chroma_dir or (out_dir / "chroma"),
    )
    print(f"  index ready in {(perf_counter() - step_started):.1f}s", flush=True)

    _demo_step(4, 6, f"Running benchmark and Pareto sweep ({options['steps']} lambda values)...")
    step_started = perf_counter()
    payload = run_benchmark(
        index_path=index_path,
        eval_suite_path=eval_path,
        token_budget=args.budget,
        conversation_budget=args.conversation_budget,
        candidate_k=args.candidate_k,
        lam=args.lam,
        cross_turn_lambda=args.cross_turn_lambda,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        use_jax=None if args.use_jax == "auto" else args.use_jax == "true",
        output_path=benchmark_path,
        pareto_steps=int(options['steps']),
        max_lam=args.max_lam,
        require_live_answer=args.require_live_answer,
        require_live_judge=args.require_live_judge,
    )
    dump_json(pareto_path, payload.get("pareto_rows", []))
    print(f"  benchmark ready in {(perf_counter() - step_started):.1f}s", flush=True)

    generated_perf = None
    if not args.skip_perf:
        _demo_step(5, 6, f"Running perf sample ({options['perf_limit']} queries, {options['perf_iterations']} iteration(s), concurrency {options['perf_concurrency']})...")
        step_started = perf_counter()
        generated_perf = run_perf(
            index_path,
            suite_path=eval_path,
            strategy='mosaic_no_ledger',
            token_budget=args.budget,
            conversation_budget=args.conversation_budget,
            candidate_k=args.candidate_k,
            lam=args.lam,
            cross_turn_lambda=args.cross_turn_lambda,
            iterations=int(options['perf_iterations']),
            warmup=int(options['perf_warmup']),
            concurrency=int(options['perf_concurrency']),
            limit=int(options['perf_limit']),
            audit_db_path=audit_db_path,
            output_path=perf_path,
        )
        print(f"  perf ready in {(perf_counter() - step_started):.1f}s", flush=True)
    else:
        _demo_step(5, 6, "Skipping perf generation.")

    _demo_step(6, 6, f"Rendering report at {report_target}...")
    step_started = perf_counter()
    report_path = render_report(
        [],
        output_path=report_target,
        title="MOSAIC Demo Report",
        benchmark_path=benchmark_path,
        audit_db_path=audit_db_path if audit_db_path.exists() else None,
        audit_export_path=args.audit_export,
        perf_path=perf_path if generated_perf is not None else None,
    )
    print(f"  report ready in {(perf_counter() - step_started):.1f}s", flush=True)

    print("Demo complete.")
    print(f"- index: {index_path}")
    if audit_db_path.exists():
        print(f"- audit_db: {audit_db_path}")
    print(f"- benchmark: {benchmark_path}")
    print(f"- pareto: {pareto_path}")
    if generated_perf is not None:
        print(f"- perf: {perf_path}")
    print(f"- report: {report_path}")
    provenance = payload.get('metadata', {}).get('evaluation_provenance', {})
    print(f"- ttft_backends: {_count_rows_to_text(provenance.get('ttft_backend_counts', []), 'backend')}")
    print(f"- faithfulness_backends: {_count_rows_to_text(provenance.get('faithfulness_backend_counts', []), 'backend')}")
    if generated_perf is not None:
        print(f"- perf_throughput_qps: {generated_perf.get('summary', {}).get('throughput_qps', 0.0)}")
    print(f"- total_runtime_s: {round(perf_counter() - started, 1)}")

    ui_command = f"mosaic ui --index {index_path}"
    if audit_db_path.exists():
        ui_command += f" --audit-db {audit_db_path}"
    ui_command += f" --report-path {report_path} --benchmark-path {benchmark_path}"
    if generated_perf is not None:
        ui_command += f" --perf-path {perf_path}"

    print("Next steps:")
    print(f'  mosaic query --index {index_path} --query "What workflow did the memo freeze?" --roles analyst --strategy mosaic_no_ledger --budget 700')
    print(f"  {ui_command}")

    if args.open_ui:
        url = f"http://{args.ui_host}:{args.ui_port}"
        print(f"Launching MOSAIC UI at {url}")
        if not args.no_open:
            _schedule_url_open(url)
            print(f"Auto-open scheduled for {url}")
        print("Press Ctrl+C to stop the UI server.")
        serve_ui(
            index_path=index_path,
            audit_db_path=audit_db_path if audit_db_path.exists() else None,
            principal_map_path=None,
            calibration_suite_path=eval_path,
            report_path=report_path,
            benchmark_path=benchmark_path,
            perf_path=perf_path if perf_path.exists() else None,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            host=args.ui_host,
            port=args.ui_port,
        )
        return 0

    if not args.no_open:
        opened = _open_path(report_path)
        print(f"Auto-open {'succeeded' if opened else 'was skipped'} for {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosaic",
        description="Permission-aware context management, auditing, and MCP serving for enterprise retrieval.",
        epilog=dedent(
            """
            Start here:
              mosaic demo --clean
              mosaic query --index .mosaic/index.json --query \"What workflow did the memo freeze?\" --roles analyst
              mosaic doctor --strict
              mosaic ui --index .mosaic/index.json --audit-db .mosaic/audit.db --report-path .mosaic/report.html
              mosaic serve-mcp --transport http --index .mosaic/index.json --audit-db .mosaic/audit.db --principal-map examples/principals.sample.json --allow-origin http://localhost
            """
        ),
        formatter_class=HelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus_parser = _add_parser(
        subparsers,
        "generate-corpus",
        help_text="Generate the 100-document synthetic corpus.",
        description="Generate the synthetic financial-services corpus used throughout the MOSAIC benchmark and demo.",
    )
    corpus_parser.add_argument("--output-dir", default="corpus/documents", help="Directory to populate with generated source documents.")
    corpus_parser.add_argument("--catalog", default="corpus/catalog.json", help="Where to write the catalog JSON describing the generated corpus.")
    corpus_parser.add_argument("--clean", action="store_true", help="Delete and regenerate the target document directory before writing new files.")
    corpus_parser.set_defaults(func=cmd_generate_corpus)

    eval_gen_parser = _add_parser(
        subparsers,
        "generate-eval",
        help_text="Generate the benchmark query suite.",
        description="Build the comprehensive evaluation suite from a corpus catalog.",
    )
    eval_gen_parser.add_argument("--catalog", default="corpus/catalog.json", help="Catalog JSON produced by `mosaic generate-corpus`.")
    eval_gen_parser.add_argument("--output", default="corpus/eval_suite.json", help="Where to write the generated evaluation suite JSON.")
    eval_gen_parser.set_defaults(func=cmd_generate_eval)

    ingest_parser = _add_parser(
        subparsers,
        "ingest",
        help_text="Chunk, embed, and index documents.",
        description="Create a MOSAIC retrieval index from a directory of source documents or a manifest-backed real corpus.",
    )
    ingest_parser.add_argument("source", nargs="?", help="Directory containing source documents to ingest. Optional when --manifest is supplied.")
    ingest_parser.add_argument("--manifest", help="Optional JSON manifest describing real corpus files, ACLs, and metadata. See examples/project_docs_manifest.json.")
    ingest_parser.add_argument("--extensions", default=",".join(SUPPORTED_SOURCE_EXTENSIONS), help="Comma-separated source file extensions to scan when ingesting a directory.")
    ingest_parser.add_argument("--output", required=True, help="Where to write the retrieval index JSON file.")
    ingest_parser.add_argument("--chunk-size", type=int, default=512, help="Target chunk size in tokens.")
    ingest_parser.add_argument("--overlap", type=int, default=64, help="Chunk overlap in tokens.")
    ingest_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name. Falls back to local hash embeddings if the model is unavailable offline.")
    ingest_parser.add_argument("--embedding-dimensions", type=int, default=384, help="Embedding dimensionality for hash-based fallback embeddings.")
    ingest_parser.add_argument("--vector-store", choices=("json", "chroma"), default="chroma", help="Backing store for embeddings and metadata.")
    ingest_parser.add_argument("--chroma-dir", help="Directory for the ChromaDB persistence layer when `--vector-store chroma` is used.")
    ingest_parser.add_argument("--collection-name", default="mosaic_chunks", help="Collection name to use inside ChromaDB.")
    ingest_parser.set_defaults(func=cmd_ingest)

    retrieve_parser = _add_parser(
        subparsers,
        "retrieve",
        help_text="Inspect retrieval candidates.",
        description="Run hybrid retrieval and print the top candidate chunks before optimization.",
    )
    _add_index_argument(retrieve_parser)
    retrieve_parser.add_argument("--query", required=True, help="Natural-language query to retrieve against the index.")
    _add_roles_argument(retrieve_parser)
    retrieve_parser.add_argument("--k", type=int, default=12, help="Number of candidates to print.")
    retrieve_parser.add_argument("--alpha", type=float, default=0.65, help="Interpolation weight between semantic and BM25 retrieval signals.")
    retrieve_parser.set_defaults(func=cmd_retrieve)

    query_parser = _add_parser(
        subparsers,
        "query",
        help_text="Run the full MOSAIC runtime query path.",
        description=dedent(
            """
            Run a live query through the canonical MosaicService path.

            Use `--budget` for single-turn or ledgerless strategies.
            Use `--conversation-budget` plus `--conversation-id` for `mosaic_full` multi-turn conversations.
            """
        ).strip(),
    )
    _add_index_argument(query_parser)
    query_parser.add_argument("--query", required=True, help="Natural-language query to answer.")
    _add_roles_argument(query_parser)
    query_parser.add_argument("--principal-id", default="local-cli", help="Principal identifier for the local caller. For HTTP, the authenticated principal overrides this.")
    _add_principal_map_argument(query_parser)
    _add_audit_db_argument(query_parser)
    query_parser.add_argument("--strategy", choices=("topk", "mmr", "mosaic_no_ledger", "mosaic_full"), default="mosaic_full", help="Selection strategy to apply after classification succeeds.")
    query_parser.add_argument("--candidate-k", type=int, default=50, help="How many retrieval candidates to pass into the optimizer or baseline selector.")
    query_parser.add_argument("--budget", type=int, default=4096, help="Per-request token budget for single-turn strategies.")
    query_parser.add_argument("--conversation-budget", type=int, default=2200, help="Total shared token budget across all turns when using `mosaic_full`.")
    query_parser.add_argument("--conversation-id", help="Reuse this ID to continue an existing multi-turn conversation ledger.")
    query_parser.add_argument("--lam", type=float, default=0.5, help="Within-turn redundancy penalty for MOSAIC optimization.")
    query_parser.add_argument("--cross-turn-lambda", type=float, default=0.75, help="Cross-turn redundancy penalty for `mosaic_full`.")
    query_parser.add_argument("--calibration-suite", help="Optional eval suite JSON used to calibrate the failure classifier thresholds.")
    query_parser.add_argument("--ledger-state", help="Optional JSON ledger snapshot for local CLI conversations. When set, this takes precedence over the SQLite-backed ledger in `--audit-db`.")
    query_parser.add_argument("--turn", type=int, help="Explicit turn number to record in the audit trace. Usually left unset.")
    query_parser.add_argument("--include-trace", action="store_true", help="Include the full trace payload in the JSON response object written by `--output`.")
    query_parser.add_argument("--output", help="Optional path to write the full response payload as JSON.")
    query_parser.set_defaults(func=cmd_query)

    optimize_parser = _add_parser(
        subparsers,
        "optimize",
        help_text="Run chunk selection without the full runtime wrapper.",
        description="Retrieve candidates and run one selection strategy under a token budget.",
    )
    _add_index_argument(optimize_parser)
    optimize_parser.add_argument("--query", required=True, help="Natural-language query to optimize for.")
    _add_roles_argument(optimize_parser)
    optimize_parser.add_argument("--strategy", choices=("topk", "mmr", "mosaic", "mosaic_no_ledger", "mosaic_full"), default="mosaic_no_ledger", help="Selection strategy to run.")
    optimize_parser.add_argument("--candidate-k", type=int, default=50, help="How many candidates to retrieve before optimization.")
    optimize_parser.add_argument("--alpha", type=float, default=0.65, help="Interpolation weight between semantic and BM25 retrieval signals.")
    optimize_parser.add_argument("--budget", type=int, default=4096, help="Token budget available to the selector.")
    optimize_parser.add_argument("--lam", type=float, default=0.5, help="Within-turn redundancy penalty.")
    optimize_parser.add_argument("--cross-turn-lambda", type=float, default=0.75, help="Cross-turn redundancy penalty used when the strategy supports it.")
    optimize_parser.add_argument("--use-jax", choices=("auto", "true", "false"), default="auto", help="Whether to force JAX usage, forbid it, or auto-detect it.")
    optimize_parser.add_argument("--output", help="Optional path to write the selected chunks as JSON.")
    optimize_parser.set_defaults(func=cmd_optimize)

    eval_parser = _add_parser(
        subparsers,
        "eval",
        help_text="Run the single-turn evaluation suite.",
        description="Benchmark one strategy over the single-turn evaluation suite.",
    )
    _add_index_argument(eval_parser)
    eval_parser.add_argument("--suite", required=True, help="Evaluation suite JSON file.")
    eval_parser.add_argument("--strategy", choices=("topk", "mmr", "mosaic", "mosaic_no_ledger", "mosaic_full"), required=True, help="Strategy to evaluate.")
    eval_parser.add_argument("--budget", type=int, default=4096, help="Token budget per query.")
    eval_parser.add_argument("--candidate-k", type=int, default=50, help="Candidate count to feed into the selector.")
    eval_parser.add_argument("--lam", type=float, default=0.5, help="Within-turn redundancy penalty for MOSAIC strategies.")
    eval_parser.add_argument("--answer-model", default=DEFAULT_ANTHROPIC_ANSWER_MODEL, help="Anthropic answer-generation model. Falls back to the local answerer when no API key is configured unless --require-live-answer is set.")
    eval_parser.add_argument("--judge-model", default=DEFAULT_ANTHROPIC_JUDGE_MODEL, help="Anthropic judge model. Falls back to proxy faithfulness metrics when no API key is configured unless --require-live-judge is set.")
    eval_parser.add_argument("--require-live-answer", action="store_true", help="Fail instead of falling back when live TTFT / answer generation is unavailable.")
    eval_parser.add_argument("--require-live-judge", action="store_true", help="Fail instead of falling back when live faithfulness judging is unavailable.")
    eval_parser.add_argument("--use-jax", choices=("auto", "true", "false"), default="auto", help="Whether to force JAX usage, forbid it, or auto-detect it.")
    eval_parser.add_argument("--output", required=True, help="Path to write the per-query evaluation rows JSON.")
    eval_parser.set_defaults(func=cmd_eval)

    pareto_parser = _add_parser(
        subparsers,
        "pareto",
        help_text="Sweep lambda and produce a Pareto frontier.",
        description="Run a lambda sweep for MOSAIC and write the quality-vs-latency frontier.",
    )
    _add_index_argument(pareto_parser)
    pareto_parser.add_argument("--suite", required=True, help="Evaluation suite JSON file.")
    pareto_parser.add_argument("--budget", type=int, default=4096, help="Token budget per query.")
    pareto_parser.add_argument("--candidate-k", type=int, default=50, help="Candidate count to feed into the selector.")
    pareto_parser.add_argument("--lam-values", help="Optional comma-separated lambda values. If omitted, values are generated from --max-lam and --steps.")
    pareto_parser.add_argument("--max-lam", type=float, default=2.0, help="Maximum lambda to include in the auto-generated sweep.")
    pareto_parser.add_argument("--steps", type=int, default=20, help="Number of lambda values to generate when --lam-values is omitted.")
    pareto_parser.add_argument("--answer-model", default=DEFAULT_ANTHROPIC_ANSWER_MODEL, help="Anthropic answer-generation model.")
    pareto_parser.add_argument("--judge-model", default=DEFAULT_ANTHROPIC_JUDGE_MODEL, help="Anthropic judge model.")
    pareto_parser.add_argument("--require-live-answer", action="store_true", help="Fail instead of falling back when live TTFT / answer generation is unavailable.")
    pareto_parser.add_argument("--require-live-judge", action="store_true", help="Fail instead of falling back when live faithfulness judging is unavailable.")
    pareto_parser.add_argument("--use-jax", choices=("auto", "true", "false"), default="auto", help="Whether to force JAX usage, forbid it, or auto-detect it.")
    pareto_parser.add_argument("--output", required=True, help="Path to write the Pareto rows JSON.")
    pareto_parser.set_defaults(func=cmd_pareto)

    benchmark_parser = _add_parser(
        subparsers,
        "benchmark",
        help_text="Run the full three-engine benchmark.",
        description="Run the comprehensive benchmark, including single-turn, classifier, multi-turn, and Pareto outputs.",
    )
    _add_index_argument(benchmark_parser)
    benchmark_parser.add_argument("--suite", required=True, help="Evaluation suite JSON file.")
    benchmark_parser.add_argument("--budget", type=int, default=4096, help="Single-turn token budget.")
    benchmark_parser.add_argument("--conversation-budget", type=int, default=2200, help="Shared multi-turn conversation budget for `mosaic_full`.")
    benchmark_parser.add_argument("--candidate-k", type=int, default=50, help="Candidate count to feed into retrieval and optimization.")
    benchmark_parser.add_argument("--lam", type=float, default=0.5, help="Within-turn redundancy penalty.")
    benchmark_parser.add_argument("--cross-turn-lambda", type=float, default=0.75, help="Cross-turn redundancy penalty.")
    benchmark_parser.add_argument("--max-lam", type=float, default=2.0, help="Maximum lambda to include in the Pareto sweep.")
    benchmark_parser.add_argument("--steps", type=int, default=20, help="Number of lambda values in the Pareto sweep.")
    benchmark_parser.add_argument("--answer-model", default=DEFAULT_ANTHROPIC_ANSWER_MODEL, help="Anthropic answer-generation model.")
    benchmark_parser.add_argument("--judge-model", default=DEFAULT_ANTHROPIC_JUDGE_MODEL, help="Anthropic judge model.")
    benchmark_parser.add_argument("--require-live-answer", action="store_true", help="Fail instead of falling back when live TTFT / answer generation is unavailable.")
    benchmark_parser.add_argument("--require-live-judge", action="store_true", help="Fail instead of falling back when live faithfulness judging is unavailable.")
    benchmark_parser.add_argument("--use-jax", choices=("auto", "true", "false"), default="auto", help="Whether to force JAX usage, forbid it, or auto-detect it.")
    benchmark_parser.add_argument("--output", required=True, help="Path to write the benchmark payload JSON.")
    benchmark_parser.set_defaults(func=cmd_benchmark)

    perf_parser = _add_parser(
        subparsers,
        "perf",
        help_text="Measure live runtime latency and throughput.",
        description="Run repeated live MOSAIC queries through the canonical runtime path and write percentile / throughput evidence as JSON.",
    )
    _add_index_argument(perf_parser)
    perf_parser.add_argument("--suite", help="Evaluation suite JSON to sample perf queries from.")
    perf_parser.add_argument("--query", action="append", help="Ad hoc query text. Repeat the flag to benchmark multiple queries.")
    _add_roles_argument(perf_parser)
    _add_principal_map_argument(perf_parser)
    perf_parser.add_argument("--audit-db", help="Optional audit DB to include persistence overhead in the perf run.")
    perf_parser.add_argument("--strategy", choices=("topk", "mmr", "mosaic_no_ledger", "mosaic_full"), default="mosaic_no_ledger", help="Runtime strategy to measure.")
    perf_parser.add_argument("--budget", type=int, default=4096, help="Per-request token budget for non-ledger strategies.")
    perf_parser.add_argument("--conversation-budget", type=int, default=2200, help="Per-request conversation budget when using `mosaic_full`.")
    perf_parser.add_argument("--candidate-k", type=int, default=50, help="How many retrieval candidates to pass into the runtime.")
    perf_parser.add_argument("--lam", type=float, default=0.5, help="Within-turn redundancy penalty.")
    perf_parser.add_argument("--cross-turn-lambda", type=float, default=0.75, help="Cross-turn redundancy penalty for `mosaic_full`.")
    perf_parser.add_argument("--iterations", type=int, default=1, help="How many times to replay the query set.")
    perf_parser.add_argument("--warmup", type=int, default=0, help="How many requests to run before measurement starts.")
    perf_parser.add_argument("--concurrency", type=int, default=1, help="How many concurrent worker threads to use.")
    perf_parser.add_argument("--limit", type=int, help="Limit the number of suite queries included in the run.")
    perf_parser.add_argument("--category", action="append", help="Restrict suite-driven perf runs to one or more categories. Repeat the flag to allow multiple categories.")
    perf_parser.add_argument("--include-multi-turn", action="store_true", help="Allow multi-turn suite items into the perf workload.")
    perf_parser.add_argument("--calibration-suite", help="Optional eval suite JSON used to calibrate the failure classifier thresholds.")
    perf_parser.add_argument("--output", required=True, help="Path to write the perf payload JSON.")
    perf_parser.set_defaults(func=cmd_perf)

    audit_parser = _add_parser(
        subparsers,
        "audit",
        help_text="Inspect MOSAIC audit traces.",
        description="Read, filter, and export the SQLite-backed audit trail written by live queries.",
    )
    audit_subparsers = audit_parser.add_subparsers(dest="audit_command", required=True)

    audit_list = _add_parser(
        audit_subparsers,
        "list",
        help_text="List recent audit events.",
        description="List recent audit requests with optional filters.",
    )
    _add_index_argument(audit_list)
    _add_audit_db_argument(audit_list)
    audit_list.add_argument("--limit", type=int, default=20, help="Maximum number of events to return.")
    audit_list.add_argument("--status", help="Filter by runtime status such as success or permission_gap.")
    audit_list.add_argument("--classification", help="Filter by classifier label such as SUCCESS or PERMISSION_GAP.")
    audit_list.add_argument("--principal-id", help="Filter by principal identifier.")
    audit_list.add_argument("--conversation-id", help="Filter by one conversation ledger identifier.")
    audit_list.add_argument("--output", help="Optional path to write the matching rows as JSON.")
    audit_list.set_defaults(func=cmd_audit_list)

    audit_show = _add_parser(
        audit_subparsers,
        "show",
        help_text="Show one full audit trace.",
        description="Fetch one complete audited request, including classifier scores, candidates, and selection factors.",
    )
    _add_index_argument(audit_show)
    _add_audit_db_argument(audit_show)
    audit_show.add_argument("--audit-id", required=True, help="Audit request identifier returned by `mosaic query` or MCP `query_context`.")
    audit_show.add_argument("--include-text", action="store_true", help="Hydrate selected chunk text into the response payload.")
    audit_show.add_argument("--candidate-limit", type=int, help="Optionally limit the number of candidate rows returned in the payload.")
    audit_show.add_argument("--summary", action="store_true", help="Return a compact summary instead of the full trace payload.")
    audit_show.add_argument("--output", help="Optional path to write the trace JSON.")
    audit_show.set_defaults(func=cmd_audit_show)

    audit_export = _add_parser(
        audit_subparsers,
        "export",
        help_text="Export audited requests as JSON.",
        description="Export one request, one conversation, or a recent batch of audited requests.",
    )
    _add_index_argument(audit_export)
    _add_audit_db_argument(audit_export)
    audit_export.add_argument("--audit-id", help="Export one audited request.")
    audit_export.add_argument("--conversation-id", help="Export all requests for one conversation.")
    audit_export.add_argument("--limit", type=int, default=50, help="Batch export size when neither --audit-id nor --conversation-id is provided.")
    audit_export.add_argument("--include-text", action="store_true", help="Hydrate selected chunk text into the export payload.")
    audit_export.add_argument("--output", help="Optional path to write the export JSON.")
    audit_export.set_defaults(func=cmd_audit_export)

    doctor_parser = _add_parser(
        subparsers,
        "doctor",
        help_text="Validate the local MOSAIC workspace.",
        description="Check that the index, audit DB, principal map, and report are present and readable.",
    )
    doctor_parser.add_argument("--index", default=".mosaic/index.json", help="Index JSON to validate.")
    doctor_parser.add_argument("--audit-db", default=".mosaic/audit.db", help="Audit DB to inspect when present.")
    doctor_parser.add_argument("--principal-map", default="examples/principals.sample.json", help="Principal map JSON to inspect when present.")
    doctor_parser.add_argument("--report-path", default=".mosaic/report.html", help="Report HTML file to inspect when present.")
    doctor_parser.add_argument("--benchmark-path", default=".mosaic/benchmark.json", help="Benchmark JSON to inspect when present.")
    doctor_parser.add_argument("--perf-path", default=".mosaic/perf.json", help="Perf JSON to inspect when present.")
    doctor_parser.add_argument("--answer-model", default=DEFAULT_ANTHROPIC_ANSWER_MODEL, help="Answer model to check for live-backend readiness.")
    doctor_parser.add_argument("--judge-model", default=DEFAULT_ANTHROPIC_JUDGE_MODEL, help="Judge model to check for live-backend readiness.")
    doctor_parser.add_argument("--require-live-answer", action="store_true", help="Fail if the live answer backend is not ready.")
    doctor_parser.add_argument("--require-live-judge", action="store_true", help="Fail if the live judge backend is not ready.")
    doctor_parser.add_argument("--strict", action="store_true", help="Treat missing optional artifacts as failures.")
    doctor_parser.add_argument("--output", help="Optional path to write the doctor payload as JSON.")
    doctor_parser.set_defaults(func=cmd_doctor)

    serve_parser = _add_parser(
        subparsers,
        "serve-mcp",
        help_text="Serve MOSAIC over MCP.",
        description=dedent(
            """
            Serve MOSAIC over MCP `stdio` or Streamable HTTP.

            For authenticated HTTP, provide a principal map with bearer tokens. See examples/principals.sample.json.
            """
        ).strip(),
    )
    serve_parser.add_argument("--transport", choices=("stdio", "http"), default="stdio", help="MCP transport to expose.")
    _add_index_argument(serve_parser)
    _add_audit_db_argument(serve_parser)
    _add_principal_map_argument(serve_parser)
    serve_parser.add_argument("--calibration-suite", help="Optional eval suite JSON used to calibrate the failure classifier thresholds.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind address for the HTTP transport.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port for the HTTP transport.")
    serve_parser.add_argument("--path", default="/mcp", help="Streamable HTTP endpoint path.")
    serve_parser.add_argument("--allow-origin", action="append", help="Allowed browser/client origins for the HTTP transport. Repeat the flag to allow multiple origins.")
    serve_parser.add_argument("--disable-http-auth", action="store_true", help="Disable bearer-token authentication for local-only HTTP development.")
    serve_parser.add_argument("--report-path", default=".mosaic/report.html", help="HTML report to expose through the `report://latest` MCP resource.")
    serve_parser.set_defaults(func=cmd_serve_mcp)

    ui_parser = _add_parser(
        subparsers,
        "ui",
        help_text="Serve the local browser UI.",
        description="Run the MOSAIC local control-plane UI for querying, auditing, workspace checks, and report review.",
    )
    _add_index_argument(ui_parser)
    _add_audit_db_argument(ui_parser)
    _add_principal_map_argument(ui_parser)
    ui_parser.add_argument("--calibration-suite", help="Optional eval suite JSON used to calibrate the failure classifier thresholds.")
    ui_parser.add_argument("--report-path", default=".mosaic/report.html", help="Report HTML to embed in the report pane.")
    ui_parser.add_argument("--benchmark-path", default=".mosaic/benchmark.json", help="Benchmark JSON used for workspace summary cards.")
    ui_parser.add_argument("--perf-path", default=".mosaic/perf.json", help="Perf JSON used for workspace summary cards.")
    ui_parser.add_argument("--answer-model", default=DEFAULT_ANTHROPIC_ANSWER_MODEL, help="Answer model shown in workspace readiness cards.")
    ui_parser.add_argument("--judge-model", default=DEFAULT_ANTHROPIC_JUDGE_MODEL, help="Judge model shown in workspace readiness cards.")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Bind address for the local UI server.")
    ui_parser.add_argument("--port", type=int, default=8090, help="Port for the local UI server.")
    ui_parser.add_argument("--open-ui", action="store_true", help="Attempt to open the local UI in the default browser.")
    ui_parser.set_defaults(func=cmd_ui)

    report_parser = _add_parser(
        subparsers,
        "report",
        help_text="Render an offline HTML report.",
        description="Render a Plotly-based HTML report from benchmark output or raw evaluation rows.",
    )
    report_parser.add_argument("--inputs", nargs="*", help="One or more raw evaluation JSON files produced by `mosaic eval`.")
    report_parser.add_argument("--benchmark", help="Comprehensive benchmark payload JSON produced by `mosaic benchmark`.")
    report_parser.add_argument("--pareto", help="Pareto frontier JSON to embed when rendering from raw eval rows.")
    report_parser.add_argument("--audit-db", help="Optional audit database to use for the governance section.")
    report_parser.add_argument("--audit-export", help="Optional audit export JSON to use for the governance section.")
    report_parser.add_argument("--perf", help="Optional perf JSON from `mosaic perf` to embed as a performance evidence section.")
    report_parser.add_argument("--output", required=True, help="Where to write the HTML report.")
    report_parser.add_argument("--title", default="MOSAIC Report", help="Report title shown in the HTML output.")
    report_parser.add_argument("--open-report", action="store_true", help="Attempt to open the generated report in the default browser.")
    report_parser.set_defaults(func=cmd_report)

    demo_parser = _add_parser(
        subparsers,
        "demo",
        help_text="Run the end-to-end local demo.",
        description="Generate the corpus, build the eval suite, ingest documents, run the benchmark, and render the report.",
    )
    demo_parser.add_argument("--output-dir", default=".mosaic", help="Directory for generated demo artifacts.")
    demo_parser.add_argument("--corpus-dir", default="corpus/documents", help="Directory where generated source documents should live.")
    demo_parser.add_argument("--catalog", default="corpus/catalog.json", help="Catalog JSON path for generated corpus metadata.")
    demo_parser.add_argument("--eval-suite", default="corpus/eval_suite.json", help="Evaluation suite JSON path.")
    demo_parser.add_argument("--clean", action="store_true", help="Regenerate the corpus directory before running the demo.")
    demo_parser.add_argument("--fast", action="store_true", help="Use the lighter local preset: hash embeddings, JSON vector store, fewer Pareto steps, and a smaller perf sample.")
    demo_parser.add_argument("--open-ui", action="store_true", help="Start the local MOSAIC UI after the demo artifacts are generated.")
    demo_parser.add_argument("--ui-host", default="127.0.0.1", help="Bind address for the UI launched by `mosaic demo --open-ui`.")
    demo_parser.add_argument("--ui-port", type=int, default=8090, help="Port for the UI launched by `mosaic demo --open-ui`.")
    demo_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens for ingestion.")
    demo_parser.add_argument("--overlap", type=int, default=64, help="Chunk overlap in tokens for ingestion.")
    demo_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    demo_parser.add_argument("--embedding-dimensions", type=int, default=384, help="Embedding dimensionality for hash fallback embeddings.")
    demo_parser.add_argument("--vector-store", choices=("json", "chroma"), default="chroma", help="Embedding store backend.")
    demo_parser.add_argument("--chroma-dir", help="Directory for the ChromaDB persistence layer.")
    demo_parser.add_argument("--budget", type=int, default=4096, help="Single-turn token budget used in the benchmark.")
    demo_parser.add_argument("--conversation-budget", type=int, default=2200, help="Shared multi-turn token budget used in the benchmark.")
    demo_parser.add_argument("--candidate-k", type=int, default=50, help="Candidate count used during retrieval and optimization.")
    demo_parser.add_argument("--lam", type=float, default=0.5, help="Within-turn redundancy penalty.")
    demo_parser.add_argument("--cross-turn-lambda", type=float, default=0.75, help="Cross-turn redundancy penalty.")
    demo_parser.add_argument("--max-lam", type=float, default=2.0, help="Maximum lambda for the Pareto sweep.")
    demo_parser.add_argument("--steps", type=int, default=20, help="Number of lambda values in the Pareto sweep.")
    demo_parser.add_argument("--answer-model", default=DEFAULT_ANTHROPIC_ANSWER_MODEL, help="Anthropic answer-generation model.")
    demo_parser.add_argument("--judge-model", default=DEFAULT_ANTHROPIC_JUDGE_MODEL, help="Anthropic judge model.")
    demo_parser.add_argument("--require-live-answer", action="store_true", help="Fail instead of falling back when live TTFT / answer generation is unavailable.")
    demo_parser.add_argument("--require-live-judge", action="store_true", help="Fail instead of falling back when live faithfulness judging is unavailable.")
    demo_parser.add_argument("--use-jax", choices=("auto", "true", "false"), default="auto", help="Whether to force JAX usage, forbid it, or auto-detect it.")
    demo_parser.add_argument("--audit-db", help="Audit database path. Defaults to <output-dir>/audit.db and is populated by the demo perf run.")
    demo_parser.add_argument("--audit-export", help="Optional audit export JSON to use when rendering a governance section in the report.")
    demo_parser.add_argument("--perf-limit", type=int, default=12, help="How many suite queries to include in the demo perf run.")
    demo_parser.add_argument("--perf-iterations", type=int, default=1, help="How many times to replay the demo perf query set.")
    demo_parser.add_argument("--perf-warmup", type=int, default=1, help="How many demo perf requests to run before measurement starts.")
    demo_parser.add_argument("--perf-concurrency", type=int, default=2, help="How many worker threads to use during the demo perf run.")
    demo_parser.add_argument("--skip-perf", action="store_true", help="Skip generating the demo perf artifact.")
    demo_parser.add_argument("--no-open", action="store_true", help="Do not auto-open the generated report when the run completes.")
    demo_parser.set_defaults(func=cmd_demo)

    return parser


def _render_exception(exc: BaseException) -> str:
    if exc.args:
        return str(exc.args[0])
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AuthError as exc:
        print(_render_exception(exc), file=sys.stderr)
        return exc.status_code
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(_render_exception(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
