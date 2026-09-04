# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio export and progress routes

"""Export rendered traces and authorise progress WebSocket streams."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import MODEL_RUN_ERROR_RESPONSES, ModelSimulateRequest
from sc_neurocore.studio.api.security import (
    _studio_identity_from_websocket_headers,
    _studio_request_id,
    _studio_websocket_accept_subprotocol,
)
from sc_neurocore.studio.models import simulate_model
from sc_neurocore.studio.platform import AuditSinkError


def build_export_router(context: StudioApiContext) -> APIRouter:
    """Build the export and progress router over shared Studio runtime state."""
    router = APIRouter()
    settings = context.settings
    studio_browser_session_manager = context.studio_browser_session_manager
    studio_policy_gateway = context.studio_policy_gateway
    studio_route_policies = context.studio_route_policies

    @router.post("/api/export/svg", responses=MODEL_RUN_ERROR_RESPONSES)
    def export_svg(req: ModelSimulateRequest) -> Any:
        """Render one catalogue model run as SVG under the fail-closed run contract."""
        from fastapi.responses import Response
        from sc_neurocore.studio.svg_export import traces_to_svg

        def fn() -> Any:
            result = simulate_model(
                name=req.name,
                param_overrides=req.params,
                dt=req.dt,
                duration=req.duration,
                current=req.current,
                protocol=req.protocol,
            )
            svg = traces_to_svg(
                time=result["time"],
                states=result["states"],
                spikes=result.get("spikes", []),
                model_name=result.get("model_name", req.name),
                dt=req.dt or 0.1,
            )
            return Response(content=svg, media_type="image/svg+xml")

        return _safe(fn)

    @router.websocket("/ws/progress")
    async def ws_progress(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin not in settings.websocket_allowed_origins:
            await websocket.close(code=1008)
            return
        if settings.enforce_route_policies:
            websocket_request_id = _studio_request_id(
                websocket.headers.get(settings.request_id_header)
            )
            websocket_policy = studio_route_policies.policy_for("WEBSOCKET", "/ws/progress")
            identity_result = _studio_identity_from_websocket_headers(
                websocket.headers,
                authenticator=context.studio_identity_authenticator,
                session_manager=studio_browser_session_manager,
                allow_header_principal=settings.allow_header_principal,
            )
            try:
                decision = studio_policy_gateway.authorize(
                    websocket_policy,
                    principal=identity_result.principal,
                    route="/ws/progress",
                    request_id=websocket_request_id,
                    identity_failure_reason=identity_result.failure_reason,
                )
            except AuditSinkError:
                await websocket.close(code=1011)
                return
            if not decision.allowed:
                await websocket.close(code=1008)
                return
        await websocket.accept(subprotocol=_studio_websocket_accept_subprotocol(websocket.headers))
        from sc_neurocore.studio.progress import ws_progress_handler

        try:
            await ws_progress_handler(websocket)
        except WebSocketDisconnect:
            pass

    return router
