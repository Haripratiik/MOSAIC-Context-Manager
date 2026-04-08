from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .utils import WILDCARD_ROLE, parse_roles, resolve_user_roles


def _normalize_roles(values: Iterable[str]) -> list[str]:
    return [str(value).strip().lower() for value in values if str(value).strip()]


class AuthError(RuntimeError):
    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class Principal:
    principal_id: str
    display_name: str
    roles: list[str]
    transport: str = "cli"
    auth_source: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_id": self.principal_id,
            "display_name": self.display_name,
            "roles": list(self.roles),
            "transport": self.transport,
            "auth_source": self.auth_source,
        }


@dataclass(slots=True)
class PrincipalResolver:
    principals: dict[str, Principal] = field(default_factory=dict)
    tokens: dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "PrincipalResolver":
        base = {
            "public": Principal("public", "Public User", ["public"], transport="stdio", auth_source="local"),
            "local-user": Principal("local-user", "Local User", ["public"], transport="stdio", auth_source="local"),
            "local-admin": Principal("local-admin", "Local Admin", [WILDCARD_ROLE], transport="stdio", auth_source="local"),
            "local-cli": Principal("local-cli", "Local CLI", [WILDCARD_ROLE], transport="cli", auth_source="local"),
            # Backward-compatible demo principals for the bundled corpus.
            "analyst": Principal("analyst", "Analyst", ["analyst"], transport="stdio", auth_source="local"),
            "senior_analyst": Principal("senior_analyst", "Senior Analyst", ["senior_analyst"], transport="stdio", auth_source="local"),
            "executive": Principal("executive", "Executive", ["executive"], transport="stdio", auth_source="local"),
            "compliance": Principal("compliance", "Compliance", ["compliance"], transport="stdio", auth_source="local"),
        }
        return cls(principals=base, tokens={})

    @classmethod
    def from_file(cls, path: str | Path | None) -> "PrincipalResolver":
        if not path:
            return cls.default()
        target = Path(path)
        if not target.exists():
            return cls.default()
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
        resolver = cls.default()
        for item in payload.get("principals", []):
            roles_value = item.get("roles", [])
            roles = _normalize_roles(roles_value) if isinstance(roles_value, list) else parse_roles(str(roles_value))
            principal = Principal(
                principal_id=str(item["principal_id"]),
                display_name=str(item.get("display_name", item["principal_id"])),
                roles=roles or ["public"],
                transport=str(item.get("transport", "http")),
                auth_source=str(item.get("auth_source", "config")),
            )
            resolver.principals[principal.principal_id] = principal
            for token in item.get("tokens", []):
                resolver.tokens[str(token)] = principal.principal_id
        for token, principal_id in payload.get("tokens", {}).items():
            resolver.tokens[str(token)] = str(principal_id)
        return resolver

    def has_bearer_tokens(self) -> bool:
        return bool(self.tokens)

    def resolve_by_id(
        self,
        principal_id: str | None,
        *,
        transport: str,
        default_roles: list[str] | None = None,
    ) -> Principal:
        if principal_id and principal_id in self.principals:
            return replace(self.principals[principal_id], transport=transport)
        roles = _normalize_roles(default_roles or [])
        if principal_id:
            fallback_roles = roles or ["public"]
            return Principal(principal_id, principal_id, fallback_roles, transport=transport, auth_source="local")
        if roles:
            joined = "-".join(roles)
            return Principal(joined, joined, roles, transport=transport, auth_source="local")
        return replace(self.principals["public"], transport=transport)

    def resolve_token(self, token: str) -> Principal:
        principal_id = self.tokens.get(token)
        if not principal_id or principal_id not in self.principals:
            raise AuthError("Invalid bearer token.", status_code=401)
        return replace(self.principals[principal_id], transport="http", auth_source="bearer")

    def resolve_stdio_roles(self, principal: Principal, requested_roles: list[str]) -> list[str]:
        if requested_roles:
            return sorted(resolve_user_roles(_normalize_roles(requested_roles)))
        return sorted(resolve_user_roles(principal.roles))

    def resolve_http_roles(self, principal: Principal, requested_roles: list[str]) -> list[str]:
        allowed_scope = resolve_user_roles(principal.roles)
        if not requested_roles:
            return sorted(allowed_scope)
        requested_exact = set(_normalize_roles(requested_roles))
        if WILDCARD_ROLE not in allowed_scope and not requested_exact.issubset(allowed_scope):
            raise AuthError("Requested roles exceed the principal's allowed scope.", status_code=403)
        return sorted(resolve_user_roles(requested_exact))


class AuthProvider:
    def authenticate(self, headers: Mapping[str, str], resolver: PrincipalResolver) -> Principal:
        raise NotImplementedError


class BearerTokenAuthProvider(AuthProvider):
    def authenticate(self, headers: Mapping[str, str], resolver: PrincipalResolver) -> Principal:
        raw = headers.get("authorization") or headers.get("Authorization")
        if not raw:
            raise AuthError("Missing bearer token.", status_code=401)
        scheme, _, token = raw.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("Malformed bearer token.", status_code=401)
        return resolver.resolve_token(token.strip())
