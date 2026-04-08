from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest

from tests.helpers import build_fixture, pick_query


class AuditCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_fixture('audit_cli')
        cls.index_path = fixture['index_path']
        cls.audit_db = fixture['audit_db']
        cls.suite = fixture['suite']

    def test_query_writes_audit_and_show_reads_trace(self) -> None:
        item = pick_query(self.suite, category='redundancy_trap')
        query_cmd = [
            sys.executable,
            '-m',
            'mosaic',
            'query',
            '--index',
            str(self.index_path),
            '--audit-db',
            str(self.audit_db),
            '--query',
            item['query'],
            '--roles',
            ','.join(item['user_roles']),
            '--strategy',
            'mosaic_full',
            '--conversation-budget',
            '900',
        ]
        query_result = subprocess.run(query_cmd, capture_output=True, text=True, check=True)
        match = re.search(r'audit_id:\s*([a-f0-9]+)', query_result.stdout)
        self.assertIsNotNone(match)
        audit_id = match.group(1)

        show_cmd = [
            sys.executable,
            '-m',
            'mosaic',
            'audit',
            'show',
            '--index',
            str(self.index_path),
            '--audit-db',
            str(self.audit_db),
            '--audit-id',
            audit_id,
        ]
        show_result = subprocess.run(show_cmd, capture_output=True, text=True, check=True)
        payload = json.loads(show_result.stdout)
        self.assertEqual(payload['audit_id'], audit_id)
        self.assertIn('selected_chunks', payload)

        summary_cmd = [
            sys.executable,
            '-m',
            'mosaic',
            'audit',
            'show',
            '--index',
            str(self.index_path),
            '--audit-db',
            str(self.audit_db),
            '--audit-id',
            audit_id,
            '--summary',
            '--candidate-limit',
            '3',
        ]
        summary_result = subprocess.run(summary_cmd, capture_output=True, text=True, check=True)
        summary_payload = json.loads(summary_result.stdout)
        self.assertEqual(summary_payload['audit_id'], audit_id)
        self.assertLessEqual(len(summary_payload['top_candidates']), 3)
        self.assertIn('candidate_count', summary_payload)


    def test_non_ledger_query_output_omits_conversation_id(self) -> None:
        item = pick_query(self.suite, category='redundancy_trap')
        query_cmd = [
            sys.executable,
            '-m',
            'mosaic',
            'query',
            '--index',
            str(self.index_path),
            '--audit-db',
            str(self.audit_db),
            '--query',
            item['query'],
            '--roles',
            ','.join(item['user_roles']),
            '--strategy',
            'mosaic_no_ledger',
            '--budget',
            '700',
        ]
        query_result = subprocess.run(query_cmd, capture_output=True, text=True, check=True)

        self.assertNotIn('conversation_id:', query_result.stdout)
        self.assertRegex(query_result.stdout, r'audit_id:\s*[a-f0-9]+')

if __name__ == '__main__':
    unittest.main()
