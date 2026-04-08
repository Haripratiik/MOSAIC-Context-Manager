from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from mosaic.corpus_builder import generate_corpus, generate_eval_suite
from mosaic.ingestor import ingest_directory, load_index
from mosaic.signal import run_signal_eval


class SignalTests(unittest.TestCase):
    def test_failure_classifier_meets_targets_on_labeled_queries(self) -> None:
        root = Path('.mosaic_test') / f"signal_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        docs_dir = root / "documents"
        catalog = generate_corpus(docs_dir, catalog_path=root / "catalog.json", clean=True)
        suite = generate_eval_suite(catalog, output_path=root / "eval_suite.json")
        ingest_directory(docs_dir, output_path=root / "index.json", chunk_size=256, overlap=32, embedding_model="hash", vector_store="json")
        index = load_index(root / "index.json")

        rows, summary, _ = run_signal_eval(index.chunks, suite, candidate_k=12, embedding_model="hash")
        self.assertEqual(len(rows), 25)
        self.assertGreaterEqual(summary["classification_accuracy"], 0.85)
        self.assertGreaterEqual(summary["type2_precision"], 0.90)


if __name__ == "__main__":
    unittest.main()