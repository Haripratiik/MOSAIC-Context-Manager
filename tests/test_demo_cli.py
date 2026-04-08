from __future__ import annotations

from pathlib import Path
import argparse
import unittest

from mosaic.cli import _demo_options, _ui_calibration_suite_path


class DemoCliTests(unittest.TestCase):
    def test_fast_demo_uses_lightweight_preset(self) -> None:
        args = argparse.Namespace(
            fast=True,
            embedding_model='all-MiniLM-L6-v2',
            vector_store='chroma',
            steps=20,
            perf_limit=12,
            perf_iterations=2,
            perf_warmup=1,
            perf_concurrency=4,
        )
        options = _demo_options(args)

        self.assertEqual(options['embedding_model'], 'hash')
        self.assertEqual(options['vector_store'], 'json')
        self.assertEqual(options['steps'], 3)
        self.assertEqual(options['perf_limit'], 2)
        self.assertEqual(options['perf_iterations'], 1)
        self.assertEqual(options['perf_warmup'], 0)
        self.assertEqual(options['perf_concurrency'], 1)


    def test_ui_calibration_suite_defaults_for_demo_index(self) -> None:
        resolved = _ui_calibration_suite_path(None, '.mosaic/index.json')
        self.assertEqual(Path(resolved), Path('corpus/eval_suite.json'))

    def test_ui_calibration_suite_does_not_force_demo_suite_for_custom_index(self) -> None:
        resolved = _ui_calibration_suite_path(None, '.mosaic/custom_index.json')
        self.assertIsNone(resolved)

    def test_non_fast_demo_preserves_requested_settings(self) -> None:
        args = argparse.Namespace(
            fast=False,
            embedding_model='all-MiniLM-L6-v2',
            vector_store='chroma',
            steps=7,
            perf_limit=9,
            perf_iterations=3,
            perf_warmup=2,
            perf_concurrency=5,
        )
        options = _demo_options(args)

        self.assertEqual(options['embedding_model'], 'all-MiniLM-L6-v2')
        self.assertEqual(options['vector_store'], 'chroma')
        self.assertEqual(options['steps'], 7)
        self.assertEqual(options['perf_limit'], 9)
        self.assertEqual(options['perf_iterations'], 3)
        self.assertEqual(options['perf_warmup'], 2)
        self.assertEqual(options['perf_concurrency'], 5)


if __name__ == '__main__':
    unittest.main()
