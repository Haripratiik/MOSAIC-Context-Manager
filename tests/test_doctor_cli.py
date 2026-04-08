from __future__ import annotations

import json
import subprocess
import sys
import unittest

from tests.helpers import build_fixture


class DoctorCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_fixture('doctor_cli')
        cls.root = fixture['root']
        cls.index_path = fixture['index_path']
        cls.audit_db = fixture['audit_db']
        cls.principal_map = cls.root / 'principals.json'
        cls.principal_map.write_text(
            '{"principals": [{"principal_id": "http-analyst", "display_name": "HTTP Analyst", "roles": ["analyst"], "transport": "http", "auth_source": "config", "tokens": ["token-analyst"]}]}',
            encoding='utf-8',
        )
        cls.report_path = cls.root / 'report.html'
        cls.report_path.write_text('<html><body>ok</body></html>', encoding='utf-8')

    def test_doctor_reports_workspace_status(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'mosaic',
                'doctor',
                '--index',
                str(self.index_path),
                '--audit-db',
                str(self.audit_db),
                '--principal-map',
                str(self.principal_map),
                '--report-path',
                str(self.report_path),
                '--strict',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload['ok'])
        self.assertGreater(payload['checks']['index']['chunks'], 0)
        self.assertEqual(payload['checks']['principal_map']['token_count'], 1)
        self.assertIn('live_eval', payload['checks'])
        self.assertTrue(payload['checks']['report']['exists'])


if __name__ == '__main__':
    unittest.main()
