from __future__ import annotations

"""MCP server surfaces for MOSAIC over stdio and Streamable HTTP."""

from contextvars import ContextVar
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
import uvicorn

from .auth import AuthError, AuthProvider, Principal, PrincipalResolver
from .service import MosaicService, QueryRequest

CURRENT_PRINCIPAL: ContextVar[Principal | None] = ContextVar("mosaic_current_principal", default=None)


class McpHttpSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, *, resolver: PrincipalResolver, auth_provider: AuthProvider | None, allowed_origins: list[str], require_auth: bool) -> None:
        super().__init__(app)
        self.resolver = resolver
        self.auth_provider = auth_provider
        self.allowed_origins = allowed_origins
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path == "/healthz":
            return await call_next(request)
        origin = request.headers.get("origin")
        if origin and self.allowed_origins and origin not in self.allowed_origins:
            return JSONResponse({"error": "Origin is not allowed."}, status_code=403)
        principal = None
        if self.require_auth:
            try:
                if self.auth_provider is None:
                    raise AuthError("HTTP auth is enabled but no auth provider is configured.", status_code=401)
                principal = self.auth_provider.authenticate(request.headers, self.resolver)
            except AuthError as exc:
                return JSONResponse({"error": str(exc)}, status_code=exc.status_code)
        token = CURRENT_PRINCIPAL.set(principal)
        try:
            return await call_next(request)
        finally:
            CURRENT_PRINCIPAL.reset(token)


def _current_principal() -> Principal | None:
    return CURRENT_PRINCIPAL.get()


def build_mcp_server(
    service: MosaicService,
    *,
    principal_resolver: PrincipalResolver,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
    allowed_origins: list[str] | None = None,
) -> FastMCP:
    server = FastMCP(
        name="MOSAIC",
        instructions="Permission-aware context management and auditing for enterprise retrieval.",
        host=host,
        port=port,
        streamable_http_path=path,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(allowed_hosts=sorted({host, f"{host}:*", "127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*"}), allowed_origins=list(allowed_origins or [])),
    )

    @server.tool(name="query_context", description="Classify a query or assemble context through MOSAIC.", structured_output=True)
    def query_context(
        query: str,
        principal_id: str | None = None,
        requested_roles: list[str] | None = None,
        strategy: str = "mosaic_full",
        candidate_k: int = 50,
        token_budget: int = 4096,
        conversation_id: str | None = None,
        turn: int | None = None,
        lam: float = 0.5,
        cross_turn_lambda: float = 0.75,
        return_trace: bool = False,
    ) -> dict[str, Any]:
        principal = _current_principal()
        response = service.query(
            QueryRequest(
                query=query,
                principal_id=principal_id,
                requested_roles=list(requested_roles or []),
                strategy=strategy,
                candidate_k=candidate_k,
                token_budget=token_budget,
                conversation_id=conversation_id,
                turn=turn,
                lam=lam,
                cross_turn_lambda=cross_turn_lambda,
                return_trace=return_trace,
            ),
            principal=principal,
            transport=principal.transport if principal is not None else "stdio",
        )
        return response.to_dict()

    @server.tool(name="get_audit_trace", description="Fetch the full audit trace for one request.", structured_output=True)
    def get_audit_trace(
        audit_id: str,
        include_text: bool = False,
        candidate_limit: int | None = None,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        return service.get_audit_trace(
            audit_id,
            include_text=include_text,
            candidate_limit=candidate_limit,
            summary_only=summary_only,
        )

    @server.tool(name="list_audit_events", description="List recent audited requests.", structured_output=True)
    def list_audit_events(
        limit: int = 20,
        status: str | None = None,
        classification: str | None = None,
        principal_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return service.list_audit_events(
            limit=limit,
            status=status,
            classification=classification,
            principal_id=principal_id,
            conversation_id=conversation_id,
        )

    @server.tool(name="export_audit_bundle", description="Export audited requests as JSON.", structured_output=True)
    def export_audit_bundle(
        audit_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
        include_text: bool = False,
    ) -> dict[str, Any]:
        return service.export_audit(audit_id=audit_id, conversation_id=conversation_id, limit=limit, include_text=include_text)

    @server.resource("audit://request/{audit_id}", name="audit-request", title="Audit Request", mime_type="application/json")
    def request_audit_resource(audit_id: str) -> dict[str, Any]:
        return service.get_audit_trace(audit_id, include_text=False)

    @server.resource("audit://conversation/{conversation_id}", name="audit-conversation", title="Audit Conversation", mime_type="application/json")
    def conversation_audit_resource(conversation_id: str) -> dict[str, Any]:
        return service.export_audit(conversation_id=conversation_id, include_text=False)

    @server.resource("report://latest", name="latest-report", title="Latest Report", mime_type="text/html")
    def latest_report_resource() -> str:
        if not service.report_path.exists():
            return "<html><body><p>No report has been generated yet.</p></body></html>"
        return service.report_path.read_text(encoding="utf-8-sig")

    return server


def build_http_app(
    server: FastMCP,
    *,
    principal_resolver: PrincipalResolver,
    auth_provider: AuthProvider | None,
    allowed_origins: list[str] | None = None,
    require_auth: bool = True,
) -> Starlette:
    app = server.streamable_http_app()
    app.router.routes.append(Route("/healthz", lambda request: PlainTextResponse("ok")))
    app.add_middleware(
        McpHttpSecurityMiddleware,
        resolver=principal_resolver,
        auth_provider=auth_provider,
        allowed_origins=list(allowed_origins or []),
        require_auth=require_auth,
    )
    return app


def serve_mcp(
    service: MosaicService,
    *,
    transport: str,
    principal_resolver: PrincipalResolver,
    auth_provider: AuthProvider | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
    allowed_origins: list[str] | None = None,
    require_auth: bool = True,
) -> None:
    server = build_mcp_server(
        service,
        principal_resolver=principal_resolver,
        host=host,
        port=port,
        path=path,
        allowed_origins=allowed_origins,
    )
    if transport == "stdio":
        server.run(transport="stdio")
        return
    app = build_http_app(
        server,
        principal_resolver=principal_resolver,
        auth_provider=auth_provider,
        allowed_origins=allowed_origins,
        require_auth=require_auth,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
