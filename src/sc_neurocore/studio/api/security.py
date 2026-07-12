# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio API security middleware

"""Enforce Studio HTTP and WebSocket identity and route policy."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from starlette.routing import Match, Route

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.platform import (
    AuditSinkError,
    Principal,
    StudioBrowserSessionManager,
    StudioIdentityAuthenticator,
    StudioIdentityResult,
)


def _studio_request_id(candidate: str | None) -> str:
    if candidate is not None:
        cleaned = candidate.strip()
        if 0 < len(cleaned) <= 128 and all(
            char.isascii() and (char.isalnum() or char in "._:-") for char in cleaned
        ):
            return cleaned
    return str(uuid4())


def _studio_timestamp_utc() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def _studio_principal_from_headers(headers: Mapping[str, str]) -> Principal | None:
    principal_id = headers.get("x-studio-principal")
    if principal_id is None or not principal_id.strip():
        return None
    raw_roles = headers.get("x-studio-roles", "")
    roles = frozenset(role.strip() for role in raw_roles.split(",") if role.strip())
    return Principal(principal_id=principal_id.strip(), roles=roles)


def _studio_identity_from_headers(
    headers: Mapping[str, str],
    *,
    authenticator: StudioIdentityAuthenticator | None,
    session_manager: StudioBrowserSessionManager,
    allow_header_principal: bool,
) -> StudioIdentityResult:
    authorization = headers.get("authorization")
    if authorization is not None and authorization.strip():
        if authenticator is None:
            session_result = session_manager.authenticate_authorization_header(authorization)
            return StudioIdentityResult(
                principal=session_result.principal,
                failure_reason=session_result.failure_reason or "invalid_identity_token",
            )
        identity_result = authenticator.authenticate_authorization_header(authorization)
        if identity_result.principal is not None:
            return identity_result
        session_result = session_manager.authenticate_authorization_header(authorization)
        if session_result.principal is not None:
            return StudioIdentityResult(
                principal=session_result.principal,
                failure_reason=session_result.failure_reason,
            )
        return identity_result
    if allow_header_principal:
        return StudioIdentityResult(principal=_studio_principal_from_headers(headers))
    return StudioIdentityResult(principal=None)


def _studio_websocket_authorization_from_headers(headers: Mapping[str, str]) -> str | None:
    authorization = headers.get("authorization")
    if authorization is not None and authorization.strip():
        return authorization
    protocols = headers.get("sec-websocket-protocol", "")
    for raw_protocol in protocols.split(","):
        protocol = raw_protocol.strip()
        if protocol.startswith("studio-bearer.") and len(protocol) > len("studio-bearer."):
            return f"Bearer {protocol.removeprefix('studio-bearer.')}"
    return None


def _studio_identity_from_websocket_headers(
    headers: Mapping[str, str],
    *,
    authenticator: StudioIdentityAuthenticator | None,
    session_manager: StudioBrowserSessionManager,
    allow_header_principal: bool,
) -> StudioIdentityResult:
    authorization = _studio_websocket_authorization_from_headers(headers)
    if authorization is not None:
        return _studio_identity_from_headers(
            {"authorization": authorization},
            authenticator=authenticator,
            session_manager=session_manager,
            allow_header_principal=False,
        )
    return _studio_identity_from_headers(
        headers,
        authenticator=authenticator,
        session_manager=session_manager,
        allow_header_principal=allow_header_principal,
    )


def _studio_websocket_accept_subprotocol(headers: Mapping[str, str]) -> str | None:
    protocols = {
        raw_protocol.strip()
        for raw_protocol in headers.get("sec-websocket-protocol", "").split(",")
        if raw_protocol.strip()
    }
    return "studio-auth" if "studio-auth" in protocols else None


def _iter_leaf_routes(routes: Iterable[Any]) -> Iterator[Route]:
    """Yield every leaf HTTP ``Route``, descending into included sub-routers.

    Starlette 1.3 stopped flattening ``include_router`` routes onto ``app.routes``
    and instead wraps each included router in an ``_IncludedRouter`` whose leaf
    routes are reachable via ``original_router.routes``. Older Starlette flattened
    them, so iterating ``app.routes`` directly sufficed. Walk both shapes so route
    policy classification cannot silently fall back to ``unclassified_route`` on a
    newer Starlette.
    """
    for route in routes:
        if isinstance(route, Route):
            yield route
        original_router = getattr(route, "original_router", None)
        if original_router is not None and hasattr(original_router, "routes"):
            yield from _iter_leaf_routes(original_router.routes)
        elif not isinstance(route, Route) and getattr(route, "routes", None):
            yield from _iter_leaf_routes(route.routes)


def _studio_route_signature(app: FastAPI, request: Request) -> tuple[str, str] | None:
    for route in _iter_leaf_routes(app.routes):
        match, _ = route.matches(request.scope)
        if match is Match.FULL:
            return request.method, route.path
    return None


def install_studio_security_middleware(
    app: FastAPI,
    context: StudioApiContext,
) -> None:
    """Install request limits, policy checks, and response security headers.

    Parameters
    ----------
    app:
        FastAPI application receiving the middleware.
    context:
        Shared runtime state used for policy and identity decisions.
    """
    settings = context.settings
    studio_browser_session_manager = context.studio_browser_session_manager
    studio_policy_gateway = context.studio_policy_gateway
    studio_route_policies = context.studio_route_policies

    @app.middleware("http")
    async def add_studio_security_headers(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        request_id = _studio_request_id(request.headers.get(settings.request_id_header))
        request.state.studio_request_id = request_id
        request.state.studio_principal = None
        content_length = request.headers.get("content-length")
        if (
            content_length is not None
            and content_length.isdecimal()
            and int(content_length) > settings.max_request_body_bytes
        ):
            response = JSONResponse(
                {"detail": "Studio request body exceeds configured limit."},
                status_code=413,
            )
        elif settings.enforce_route_policies:
            route_signature = _studio_route_signature(app, request)
            if route_signature is None:
                response = JSONResponse({"detail": "unclassified_route"}, status_code=403)
            else:
                method, path_template = route_signature
                policy = studio_route_policies.policy_for(method, path_template)
                identity_result = _studio_identity_from_headers(
                    request.headers,
                    authenticator=context.studio_identity_authenticator,
                    session_manager=studio_browser_session_manager,
                    allow_header_principal=settings.allow_header_principal,
                )
                try:
                    decision = studio_policy_gateway.authorize(
                        policy,
                        principal=identity_result.principal,
                        route=path_template,
                        request_id=request_id,
                        identity_failure_reason=identity_result.failure_reason,
                    )
                    if decision.allowed:
                        request.state.studio_principal = identity_result.principal
                        response = await call_next(request)
                    else:
                        response = JSONResponse(
                            {"detail": decision.reason},
                            status_code=decision.status_code,
                        )
                except AuditSinkError:
                    response = JSONResponse(
                        {"detail": "audit_append_failed"},
                        status_code=503,
                    )
        else:
            response = await call_next(request)
        for name, value in settings.http_security_headers.items():
            response.headers.setdefault(name, value)
        response.headers.setdefault(settings.request_id_header, request_id)
        return response
