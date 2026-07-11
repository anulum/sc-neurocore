# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio system and capability routes

"""Expose Studio health, capability, and operator-status adapters."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.platform import build_studio_operator_status


def build_system_router(context: StudioApiContext) -> APIRouter:
    """Build the system and capability router over shared Studio runtime state."""
    router = APIRouter()
    settings = context.settings
    studio_audit_sink = context.studio_audit_sink
    studio_browser_login_throttle = context.studio_browser_login_throttle
    studio_capabilities = context.studio_capabilities
    studio_job_manager = context.studio_job_manager
    studio_route_policies = context.studio_route_policies

    @router.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    @router.get("/api/studio/capabilities")
    def api_studio_capabilities() -> dict[str, list[dict[str, object]]]:
        return {
            "capabilities": [
                capability.to_public_dict() for capability in studio_capabilities.health_all()
            ]
        }

    @router.get("/api/studio/capabilities/{capability_id}")
    def api_studio_capability(capability_id: str) -> dict[str, object]:
        try:
            return studio_capabilities.health(capability_id).to_public_dict()
        except KeyError as exc:
            raise HTTPException(404, f"Capability '{capability_id}' not found") from exc

    @router.get("/api/studio/operator/status")
    def api_studio_operator_status() -> dict[str, object]:
        """Return aggregate, path-free Studio operator control-plane health."""
        return build_studio_operator_status(
            settings=settings,
            capabilities=tuple(studio_capabilities.health_all()),
            audit_status=studio_audit_sink.status(),
            browser_login_snapshot=studio_browser_login_throttle.snapshot(),
            job_status=studio_job_manager.status(),
            route_policy_registry=studio_route_policies,
        ).to_public_dict()

    return router
