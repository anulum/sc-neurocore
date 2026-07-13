# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training and workspace route policies

"""Training and workspace route-policy catalogue."""

from __future__ import annotations

from sc_neurocore.studio.platform.policy_models import RouteVisibility

WORKSPACE_ROUTES: tuple[tuple[str, str, RouteVisibility, str], ...] = (
    (
        "GET",
        "/api/training/jobs",
        RouteVisibility.AUTHENTICATED,
        "studio.training.jobs.read",
    ),
    (
        "GET",
        "/api/training/status/{job_id}",
        RouteVisibility.AUTHENTICATED,
        "studio.training.status.read",
    ),
    (
        "GET",
        "/api/training/checkpoint/{job_id}",
        RouteVisibility.AUTHENTICATED,
        "studio.training.checkpoint.export",
    ),
    (
        "POST",
        "/api/training/checkpoint/import",
        RouteVisibility.AUTHENTICATED,
        "studio.training.checkpoint.import",
    ),
    (
        "GET",
        "/api/training/stream/{job_id}",
        RouteVisibility.AUTHENTICATED,
        "studio.training.stream",
    ),
    (
        "POST",
        "/api/training/start",
        RouteVisibility.AUTHENTICATED,
        "studio.training.start",
    ),
    (
        "POST",
        "/api/training/stop",
        RouteVisibility.AUTHENTICATED,
        "studio.training.stop",
    ),
    (
        "GET",
        "/api/project/list",
        RouteVisibility.AUTHENTICATED,
        "studio.project.list",
    ),
    (
        "GET",
        "/api/project/load/{name}",
        RouteVisibility.AUTHENTICATED,
        "studio.project.load",
    ),
    (
        "POST",
        "/api/project/save",
        RouteVisibility.AUTHENTICATED,
        "studio.project.save",
    ),
    (
        "DELETE",
        "/api/project/{name}",
        RouteVisibility.AUTHENTICATED,
        "studio.project.delete",
    ),
    (
        "POST",
        "/api/pipeline/run",
        RouteVisibility.AUTHENTICATED,
        "studio.pipeline.run",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/actions/{action_id}/resolve",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.actions.resolve",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/actions/{action_id}/execute",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.actions.execute",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/actions/execute-all",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.actions.execute_all",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/default-flow/run",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.default_flow.run",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/default-flow/verify",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.default_flow.verify",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/default-flow/run-guarded",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.default_flow.run_guarded",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/default-flow/run-from-contract",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.default_flow.run_from_contract",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/default-flow/attest",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.default_flow.attest",
    ),
    (
        "POST",
        "/api/presets/{preset_id}/default-flow/attest/verify",
        RouteVisibility.AUTHENTICATED,
        "studio.presets.default_flow.attest_verify",
    ),
    (
        "WEBSOCKET",
        "/ws/progress",
        RouteVisibility.AUTHENTICATED,
        "studio.websocket.progress",
    ),
)
