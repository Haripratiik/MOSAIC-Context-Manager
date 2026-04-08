from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from mosaic.evaluator import aggregate_results, run_eval
from tests.helpers import build_fixture


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_fixture('evaluator')
        cls.index_path = fixture['index_path']
        cls.eval_path = fixture['eval_path']

    def test_eval_rows_include_backend_provenance(self) -> None:
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}, clear=False), patch('mosaic.evaluator.anthropic', None):
            rows = run_eval(
                self.index_path,
                self.eval_path,
                strategy='topk',
                token_budget=600,
                candidate_k=10,
                lam=0.5,
                use_jax=False,
            )

        self.assertGreater(len(rows), 0)
        self.assertIn('ttft_backend', rows[0])
        self.assertIn('ttft_backend_reason', rows[0])
        self.assertIn('fai_backend', rows[0])
        self.assertIn('fai_backend_reason', rows[0])
        summary = aggregate_results(rows)
        provenance = summary['backend_provenance']
        self.assertGreater(provenance['proxy_ttft_rows'], 0)
        self.assertGreater(provenance['proxy_faithfulness_rows'], 0)

    def test_require_live_judge_raises_without_credentials(self) -> None:
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}, clear=False), patch('mosaic.evaluator.anthropic', None):
            with self.assertRaises(RuntimeError):
                run_eval(
                    self.index_path,
                    self.eval_path,
                    strategy='topk',
                    token_budget=600,
                    candidate_k=10,
                    lam=0.5,
                    use_jax=False,
                    require_live_judge=True,
                )


if __name__ == '__main__':
    unittest.main()
