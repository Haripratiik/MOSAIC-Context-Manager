from __future__ import annotations

import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

import anyio
import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from mosaic.mcp_server import build_mcp_server
from tests.helpers import build_fixture, pick_query


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise TimeoutError(f'Server did not become ready: {url}')


class McpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_fixture('mcp')
        cls.index_path = fixture['index_path']
        cls.audit_db = fixture['audit_db']
        cls.suite = fixture['suite']
        cls.service = fixture['service']
        cls.resolver = fixture['resolver']
        cls.success_item = pick_query(cls.suite, category='redundancy_trap')
        cls.principal_map = fixture['root'] / 'principals.json'
        cls.principal_map.write_text(
            '{"principals": [{"principal_id": "http-analyst", "display_name": "HTTP Analyst", "roles": ["analyst"], "transport": "http", "auth_source": "config", "tokens": ["token-analyst"]}]}',
            encoding='utf-8',
        )

    def test_stdio_server_exposes_tools_and_resources(self) -> None:
        async def scenario() -> None:
            server = build_mcp_server(self.service, principal_resolver=self.resolver)
            tool_names = {tool.name for tool in await server.list_tools()}
            self.assertIn('query_context', tool_names)
            self.assertIn('get_audit_trace', tool_names)
            resource_uris = {str(resource.uri) for resource in await server.list_resources()}
            self.assertIn('report://latest', resource_uris)
            template_uris = {template.uriTemplate for template in await server.list_resource_templates()}
            self.assertIn('audit://request/{audit_id}', template_uris)

        anyio.run(scenario)

    def test_http_server_requires_tokens_when_auth_is_enabled(self) -> None:
        port = _free_port()
        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'mosaic',
                'serve-mcp',
                '--transport',
                'http',
                '--index',
                str(self.index_path),
                '--audit-db',
                str(self.audit_db),
                '--host',
                '127.0.0.1',
                '--port',
                str(port),
            ],
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn('no bearer tokens are configured', result.stderr.lower())

    def test_http_server_auth_and_origin_guards(self) -> None:
        port = _free_port()
        proc = subprocess.Popen(
            [
                sys.executable,
                '-m',
                'mosaic',
                'serve-mcp',
                '--transport',
                'http',
                '--index',
                str(self.index_path),
                '--audit-db',
                str(self.audit_db),
                '--principal-map',
                str(self.principal_map),
                '--host',
                '127.0.0.1',
                '--port',
                str(port),
                '--allow-origin',
                'http://localhost',
            ],
            cwd=str(Path.cwd()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_http(f'http://127.0.0.1:{port}/healthz')
            missing = httpx.post(f'http://127.0.0.1:{port}/mcp', json={}, timeout=2.0)
            self.assertEqual(missing.status_code, 401)
            bad_origin = httpx.post(
                f'http://127.0.0.1:{port}/mcp',
                headers={'Authorization': 'Bearer token-analyst', 'Origin': 'http://evil.test'},
                json={},
                timeout=2.0,
            )
            self.assertEqual(bad_origin.status_code, 403)

            async def scenario() -> None:
                async with httpx.AsyncClient(headers={'Authorization': 'Bearer token-analyst'}) as http_client:
                    async with streamable_http_client(
                        f'http://127.0.0.1:{port}/mcp',
                        http_client=http_client,
                    ) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.call_tool(
                                'query_context',
                                {
                                    'query': self.success_item['query'],
                                    'requested_roles': ['analyst'],
                                    'strategy': 'mosaic_full',
                                    'token_budget': 900,
                                },
                            )
                            payload = result.structuredContent
                            self.assertEqual(payload['status'], 'success')
                            self.assertTrue(payload['audit_id'])

                            trace_result = await session.call_tool(
                                'get_audit_trace',
                                {
                                    'audit_id': payload['audit_id'],
                                    'summary_only': True,
                                    'candidate_limit': 3,
                                },
                            )
                            trace_payload = trace_result.structuredContent
                            self.assertEqual(trace_payload['audit_id'], payload['audit_id'])
                            self.assertLessEqual(len(trace_payload['top_candidates']), 3)

            anyio.run(scenario)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == '__main__':
    unittest.main()
