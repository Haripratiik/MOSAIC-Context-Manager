from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from mosaic.audit import SQLiteAuditStore, SQLiteLedgerStore
from mosaic.auth import Principal, PrincipalResolver
from mosaic.corpus_builder import generate_corpus, generate_eval_suite
from mosaic.ingestor import ingest_directory
from mosaic.service import MosaicService
from mosaic.utils import load_json


def build_fixture(name: str) -> dict:
    root = Path('.mosaic_test') / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    docs_dir = root / 'documents'
    catalog_path = root / 'catalog.json'
    eval_path = root / 'eval_suite.json'
    index_path = root / 'index.json'
    audit_db = root / 'audit.db'
    catalog = generate_corpus(docs_dir, catalog_path=catalog_path, clean=True)
    suite = generate_eval_suite(catalog, output_path=eval_path)
    ingest_directory(docs_dir, output_path=index_path, chunk_size=256, overlap=32, embedding_model='hash', vector_store='json')
    resolver = PrincipalResolver.default()
    resolver.principals['http-analyst'] = Principal('http-analyst', 'HTTP Analyst', ['analyst'], transport='http', auth_source='config')
    resolver.tokens['token-analyst'] = 'http-analyst'
    service = MosaicService(
        index_path,
        audit_store=SQLiteAuditStore(audit_db),
        ledger_store=SQLiteLedgerStore(audit_db),
        principal_resolver=resolver,
        calibration_suite_path=eval_path,
        report_path=root / 'report.html',
    )
    return {
        'root': root,
        'docs_dir': docs_dir,
        'catalog_path': catalog_path,
        'eval_path': eval_path,
        'index_path': index_path,
        'audit_db': audit_db,
        'suite': suite,
        'resolver': resolver,
        'service': service,
    }


def pick_query(suite: object, *, category: str, ground_truth_type: str | None = None) -> dict:
    queries = suite['queries'] if isinstance(suite, dict) and 'queries' in suite else list(suite)
    for item in queries:
        if item.get('category') != category:
            continue
        if ground_truth_type is not None and item.get('ground_truth_type') != ground_truth_type:
            continue
        return item
    raise KeyError(f'No query found for {category} / {ground_truth_type}')
