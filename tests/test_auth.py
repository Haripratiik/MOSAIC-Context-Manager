from __future__ import annotations

import unittest

from mosaic.auth import AuthError, BearerTokenAuthProvider
from tests.helpers import build_fixture


class AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_fixture('auth')
        cls.resolver = fixture['resolver']

    def test_resolver_reports_token_support(self) -> None:
        self.assertTrue(self.resolver.has_bearer_tokens())

    def test_bearer_auth_accepts_valid_token(self) -> None:
        provider = BearerTokenAuthProvider()
        principal = provider.authenticate({'Authorization': 'Bearer token-analyst'}, self.resolver)
        self.assertEqual(principal.principal_id, 'http-analyst')
        self.assertEqual(principal.transport, 'http')

    def test_bearer_auth_rejects_invalid_token(self) -> None:
        provider = BearerTokenAuthProvider()
        with self.assertRaises(AuthError) as exc:
            provider.authenticate({'Authorization': 'Bearer bad-token'}, self.resolver)
        self.assertEqual(exc.exception.status_code, 401)

    def test_http_role_intersection_and_insufficient_permission(self) -> None:
        principal = self.resolver.resolve_token('token-analyst')
        narrowed = self.resolver.resolve_http_roles(principal, ['public'])
        self.assertEqual(narrowed, ['public'])
        with self.assertRaises(AuthError) as exc:
            self.resolver.resolve_http_roles(principal, ['executive'])
        self.assertEqual(exc.exception.status_code, 403)



    def test_http_wildcard_principal_can_request_arbitrary_roles(self) -> None:
        principal = self.resolver.resolve_by_id('local-admin', transport='http')
        narrowed = self.resolver.resolve_http_roles(principal, ['finance'])
        self.assertEqual(narrowed, ['finance', 'public'])

if __name__ == '__main__':
    unittest.main()
