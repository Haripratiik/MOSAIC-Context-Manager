from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from uuid import uuid4

from mosaic.corpus_builder import generate_corpus, generate_eval_suite


class CorpusBuilderTests(unittest.TestCase):
    def test_corpus_and_eval_sizes_match_comprehensive_plan(self) -> None:
        root = Path('.mosaic_test') / f"corpus_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        catalog = generate_corpus(root / "documents", catalog_path=root / "catalog.json", clean=True)
        self.assertEqual(len(catalog), 100)
        counts = Counter(item["doc_type"] for item in catalog)
        self.assertEqual(counts["earnings_report"], 40)
        self.assertEqual(counts["risk_disclosure"], 20)
        self.assertEqual(counts["compliance_memo"], 20)
        self.assertEqual(counts["research_note"], 10)
        self.assertEqual(counts["executive_briefing"], 10)

        suite = generate_eval_suite(catalog, output_path=root / "eval_suite.json")
        self.assertEqual(suite["metadata"]["total_queries"], 120)
        categories = Counter(item["category"] for item in suite["queries"])
        self.assertEqual(categories["redundancy_trap"], 25)
        self.assertEqual(categories["multi_hop"], 20)
        self.assertEqual(categories["failure_classification"], 25)
        self.assertEqual(categories["multi_turn"], 50)


if __name__ == "__main__":
    unittest.main()