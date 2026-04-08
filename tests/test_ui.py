from __future__ import annotations

import json
import unittest

from starlette.testclient import TestClient

from mosaic.ui import build_ui_app
from tests.helpers import build_fixture, pick_query


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_fixture('ui')
        cls.root = fixture['root']
        cls.index_path = fixture['index_path']
        cls.audit_db = fixture['audit_db']
        cls.eval_path = fixture['eval_path']
        cls.principal_map = fixture['root'] / 'principals.json'
        cls.principal_map.write_text(
            json.dumps(
                {
                    'principals': [
                        {
                            'principal_id': 'local-ui',
                            'display_name': 'Local UI',
                            'roles': ['public', 'analyst'],
                            'transport': 'stdio',
                            'auth_source': 'config',
                        }
                    ]
                }
            ),
            encoding='utf-8',
        )
        cls.report_path = fixture['root'] / 'report.html'
        cls.report_path.write_text('<html><body><h1>UI Report</h1></body></html>', encoding='utf-8')
        cls.benchmark_path = fixture['root'] / 'benchmark.json'
        cls.benchmark_path.write_text(
            json.dumps(
                {
                    'metadata': {'evaluation_provenance': {'all_live': False, 'ttft_backend_counts': [], 'faithfulness_backend_counts': []}},
                    'summary': {'strategy_metrics': [], 'category_metrics': [], 'classification': {}},
                    'single_turn_rows': [],
                    'failure_rows': [],
                    'multi_turn_rows': [],
                    'pareto_rows': [],
                }
            ),
            encoding='utf-8',
        )
        cls.perf_path = fixture['root'] / 'perf.json'
        cls.perf_path.write_text(
            json.dumps(
                {
                    'metadata': {'strategy': 'mosaic_no_ledger', 'categories': ['redundancy_trap'], 'concurrency': 1},
                    'summary': {'request_count': 1, 'throughput_qps': 1.23, 'latency_ms': {'total': {'p95': 42.0}, 'retrieval': {'p95': 10.0}, 'optimization': {'p95': 12.0}}, 'status_counts': [{'status': 'success', 'count': 1}]},
                }
            ),
            encoding='utf-8',
        )
        cls.query_item = pick_query(fixture['suite'], category='redundancy_trap')

    def test_ui_endpoints_cover_query_audit_and_report(self) -> None:
        app = build_ui_app(
            index_path=self.index_path,
            audit_db_path=self.audit_db,
            principal_map_path=self.principal_map,
            calibration_suite_path=self.eval_path,
            report_path=self.report_path,
            benchmark_path=self.benchmark_path,
            perf_path=self.perf_path,
            answer_model='claude-haiku-4-5-20251001',
            judge_model='claude-haiku-4-5-20251001',
        )
        client = TestClient(app)

        home = client.get('/')
        self.assertEqual(home.status_code, 200)
        self.assertIn('Mosaic Control Plane', home.text)
        self.assertIn('Loaded Documents', home.text)
        self.assertIn('Primary Output', home.text)
        self.assertIn('This pane is the answer output', home.text)
        self.assertIn('Permission Labels', home.text)
        self.assertIn('permission labels', home.text)
        self.assertIn('Click a label to replace the current permission scope', home.text)
        self.assertIn('Show raw details', home.text)
        self.assertIn(r"replace(/\n/g, '<br>');", home.text)
        self.assertEqual(home.headers.get('cache-control'), 'no-store')
        self.assertIn('function resizeReportFrame()', home.text)

        workspace = client.get('/api/workspace')
        self.assertEqual(workspace.status_code, 200)
        workspace_payload = workspace.json()
        self.assertTrue(workspace_payload['checks']['benchmark']['exists'])
        self.assertTrue(workspace_payload['checks']['perf']['exists'])
        initial_request_count = workspace_payload['checks']['audit_db'].get('request_count', 0)

        documents = client.get('/api/documents')
        self.assertEqual(documents.status_code, 200)
        documents_payload = documents.json()
        self.assertGreaterEqual(len(documents_payload), 1)
        self.assertIn('document_id', documents_payload[0])
        self.assertIn('preview', documents_payload[0])

        query = client.post(
            '/api/query',
            json={
                'query': self.query_item['query'],
                'requested_roles': self.query_item['user_roles'],
                'strategy': 'mosaic_full',
                'token_budget': 900,
                'candidate_k': 20,
            },
        )
        self.assertEqual(query.status_code, 200)
        query_payload = query.json()
        self.assertTrue(query_payload['audit_id'])

        workspace_after = client.get('/api/workspace')
        self.assertEqual(workspace_after.status_code, 200)
        self.assertEqual(workspace_after.json()['checks']['audit_db'].get('request_count', 0), initial_request_count + 1)

        permission_gap = client.post(
            '/api/query',
            json={
                'query': 'What workflow did the operations review compliance memo place under a temporary freeze for Acadia Securities?',
                'requested_roles': ['analyst'],
                'strategy': 'mosaic_no_ledger',
                'token_budget': 900,
                'candidate_k': 50,
            },
        )
        self.assertEqual(permission_gap.status_code, 200)
        permission_payload = permission_gap.json()
        self.assertEqual(permission_payload['status'], 'permission_gap')
        self.assertEqual(permission_payload['classification'], 'PERMISSION_GAP')
        self.assertEqual(permission_payload['required_role'], 'compliance')

        events = client.get('/api/audit/events?limit=5')
        self.assertEqual(events.status_code, 200)
        events_payload = events.json()
        self.assertGreaterEqual(len(events_payload), 1)

        trace = client.get(f"/api/audit/trace?audit_id={query_payload['audit_id']}&summary_only=1&candidate_limit=5")
        self.assertEqual(trace.status_code, 200)
        self.assertEqual(trace.json()['audit_id'], query_payload['audit_id'])

        report = client.get('/report')
        self.assertEqual(report.status_code, 200)
        self.assertIn('UI Report', report.text)


if __name__ == '__main__':
    unittest.main()
