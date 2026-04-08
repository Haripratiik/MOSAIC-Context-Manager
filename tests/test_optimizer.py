from __future__ import annotations

import unittest

from mosaic.optimizer import optimize, select_topk
from mosaic.types import Chunk


def make_chunk(
    chunk_id: str,
    embedding: list[float],
    relevance: float,
    token_count: int = 10,
    roles: list[str] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=chunk_id.split("::")[0],
        text=chunk_id,
        embedding=embedding,
        token_count=token_count,
        roles=roles or ["public"],
        relevance=relevance,
    )


class OptimizerTests(unittest.TestCase):
    def test_budget_never_exceeded(self) -> None:
        chunks = [
            make_chunk("a::0", [1.0, 0.0], relevance=0.9, token_count=70),
            make_chunk("b::0", [0.9, 0.1], relevance=0.8, token_count=70),
            make_chunk("c::0", [0.0, 1.0], relevance=0.7, token_count=40),
        ]
        result = optimize(chunks, user_roles=["public"], token_budget=100, lam=0.5, use_jax=False)
        self.assertLessEqual(result.tokens_used, 100)
        self.assertEqual(result.backend, "numpy")

    def test_permission_zero_chunks_never_selected(self) -> None:
        chunks = [
            make_chunk("public::0", [1.0, 0.0], relevance=0.9, roles=["public"]),
            make_chunk("private::0", [1.0, 0.0], relevance=1.0, roles=["executive"]),
        ]
        result = optimize(chunks, user_roles=["public"], token_budget=20, lam=0.5, use_jax=False)
        self.assertEqual([chunk.id for chunk in result.selected], ["public::0"])

    def test_lam_zero_matches_top_relevance_selection(self) -> None:
        chunks = [
            make_chunk("a::0", [1.0, 0.0], relevance=0.95, token_count=50),
            make_chunk("b::0", [0.9, 0.1], relevance=0.75, token_count=50),
            make_chunk("c::0", [0.0, 1.0], relevance=0.70, token_count=50),
        ]
        baseline = select_topk(chunks, token_budget=100, user_roles=["public"])
        result = optimize(chunks, user_roles=["public"], token_budget=100, lam=0.0, use_jax=False)
        self.assertEqual([chunk.id for chunk in result.selected], [chunk.id for chunk in baseline])

    def test_high_lambda_prefers_diverse_chunks(self) -> None:
        chunks = [
            make_chunk("dup_a::0", [1.0, 0.0], relevance=1.0),
            make_chunk("dup_b::0", [1.0, 0.0], relevance=0.99),
            make_chunk("novel::0", [0.0, 1.0], relevance=0.75),
        ]
        result = optimize(chunks, user_roles=["public"], token_budget=20, lam=3.0, lr=0.15, steps=120, use_jax=False)
        selected_ids = {chunk.id for chunk in result.selected}
        self.assertIn("dup_a::0", selected_ids)
        self.assertIn("novel::0", selected_ids)
        self.assertNotIn("dup_b::0", selected_ids)


if __name__ == "__main__":
    unittest.main()
