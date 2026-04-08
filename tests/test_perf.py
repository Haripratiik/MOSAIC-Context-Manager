from __future__ import annotations

import unittest

from mosaic.perf import run_perf
from tests.helpers import build_fixture


class PerfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_fixture('perf')
        cls.root = fixture['root']
        cls.index_path = fixture['index_path']
        cls.eval_path = fixture['eval_path']

    def test_run_perf_writes_latency_and_throughput_summary(self) -> None:
        output_path = self.root / 'perf.json'
        payload = run_perf(
            self.index_path,
            suite_path=self.eval_path,
            strategy='mosaic_no_ledger',
            token_budget=600,
            candidate_k=10,
            iterations=1,
            warmup=1,
            concurrency=2,
            limit=4,
            output_path=output_path,
        )

        self.assertTrue(output_path.exists())
        self.assertEqual(payload['summary']['request_count'], 3)
        self.assertGreater(payload['summary']['throughput_qps'], 0.0)
        self.assertGreater(payload['summary']['latency_ms']['total']['p95'], 0.0)
        self.assertGreater(len(payload['samples']), 0)


if __name__ == '__main__':
    unittest.main()
