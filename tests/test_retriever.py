from __future__ import annotations

import unittest

from mosaic.retriever import hybrid_retrieve
from mosaic.types import Chunk


class RetrieverTests(unittest.TestCase):
    def test_permission_prefilter(self) -> None:
        chunks = [
            Chunk("public::0", "public", "market risk overview", [1.0, 0.0], 5, ["public"]),
            Chunk("private::0", "private", "internal compliance issue", [0.0, 1.0], 5, ["compliance"]),
        ]
        results = hybrid_retrieve("compliance issue", chunks, user_roles=["public"], k=5, embedding_model="hash")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "public::0")


if __name__ == "__main__":
    unittest.main()
