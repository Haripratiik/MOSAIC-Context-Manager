from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from mosaic.corpus_builder import generate_corpus, generate_eval_suite
from mosaic.evaluator import aggregate_multiturn, run_multiturn_eval
from mosaic.ingestor import ingest_directory
from mosaic.ledger import ContextLedger
from mosaic.types import Chunk


class LedgerTests(unittest.TestCase):
    def test_context_ledger_tracks_budget_and_similarity(self) -> None:
        ledger = ContextLedger(total_budget=100)
        chunks = [
            Chunk("a::0", "a", "alpha", [1.0, 0.0], 30, ["public"]),
            Chunk("b::0", "b", "beta", [0.0, 1.0], 20, ["public"]),
        ]
        ledger.add_chunks([chunks[0]], turn=1, query_id="q1")
        self.assertEqual(ledger.remaining_budget, 70)
        sims = ledger.cross_turn_similarity([chunks[0].embedding, chunks[1].embedding])
        self.assertEqual(len(sims), 2)
        self.assertGreaterEqual(sims[0], sims[1])

    def test_mosaic_full_reduces_cross_turn_redundancy(self) -> None:
        root = Path('.mosaic_test') / f"ledger_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        docs_dir = root / "documents"
        catalog = generate_corpus(docs_dir, catalog_path=root / "catalog.json", clean=True)
        generate_eval_suite(catalog, output_path=root / "eval_suite.json")
        ingest_directory(docs_dir, output_path=root / "index.json", chunk_size=256, overlap=32, embedding_model="hash", vector_store="json")

        no_ledger_rows = run_multiturn_eval(root / "index.json", root / "eval_suite.json", strategy="mosaic_no_ledger", conversation_budget=900, candidate_k=12, lam=0.5, cross_turn_lambda=0.75, use_jax=False)
        full_rows = run_multiturn_eval(root / "index.json", root / "eval_suite.json", strategy="mosaic_full", conversation_budget=900, candidate_k=12, lam=0.5, cross_turn_lambda=0.75, use_jax=False)
        no_ledger_summary = aggregate_multiturn(no_ledger_rows)
        full_summary = aggregate_multiturn(full_rows)
        no_ledger_xtr = next(row["cross_turn_redundancy"] for row in no_ledger_summary["strategies"] if row["strategy"] == "mosaic_no_ledger")
        full_xtr = next(row["cross_turn_redundancy"] for row in full_summary["strategies"] if row["strategy"] == "mosaic_full")
        self.assertLessEqual(full_xtr, no_ledger_xtr)


if __name__ == "__main__":
    unittest.main()