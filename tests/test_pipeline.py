from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from mosaic.corpus_builder import generate_corpus, generate_eval_suite
from mosaic.evaluator import run_benchmark
from mosaic.ingestor import ingest_directory
from mosaic.perf import run_perf
from mosaic.report import render_report


class PipelineTests(unittest.TestCase):
    def test_benchmark_and_report_pipeline(self) -> None:
        root = Path('.mosaic_test') / f"pipeline_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        docs_dir = root / "documents"
        catalog_path = root / "catalog.json"
        eval_path = root / "eval_suite.json"
        index_path = root / "index.json"
        benchmark_path = root / "benchmark.json"

        catalog = generate_corpus(docs_dir, catalog_path=catalog_path, clean=True)
        generate_eval_suite(catalog, output_path=eval_path)
        ingest_directory(docs_dir, output_path=index_path, chunk_size=256, overlap=32, embedding_model="hash", vector_store="json")

        payload = run_benchmark(index_path, eval_path, token_budget=600, conversation_budget=900, candidate_k=10, lam=0.5, cross_turn_lambda=0.75, use_jax=False, output_path=benchmark_path, pareto_steps=3, max_lam=1.0)
        self.assertEqual(len(payload["failure_rows"]), 25)
        self.assertEqual(len(payload["pareto_rows"]), 3)
        self.assertTrue(benchmark_path.exists())

        perf_path = root / "perf.json"
        run_perf(index_path, suite_path=eval_path, strategy='mosaic_no_ledger', token_budget=600, candidate_k=10, limit=3, output_path=perf_path)

        report_path = render_report([], output_path=root / "report.html", title="Test MOSAIC Report", benchmark_path=benchmark_path, perf_path=perf_path)
        self.assertTrue(report_path.exists())
        report_html = report_path.read_text(encoding="utf-8-sig")
        self.assertIn("Test MOSAIC Report", report_html)
        self.assertIn("Performance Evidence", report_html)
        self.assertIn("Chunk Legend", report_html)


if __name__ == "__main__":
    unittest.main()