# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy route catalogue tests

"""Route catalogue tests for Studio policy."""

from __future__ import annotations

import pytest

from tests.studio_policy_support import policy_contract


def test_route_policy_registry_rejects_duplicate_method_path() -> None:
    contract = policy_contract()
    registry = contract["RoutePolicyRegistry"]()
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].PUBLIC,
        audit_action="studio.health.read",
    )
    registry.register("GET", "/api/health", policy)

    with pytest.raises(ValueError, match="already has a Studio route policy"):
        registry.register("get", "/api/health", policy)


def test_default_route_policy_registry_classifies_platform_routes() -> None:
    contract = policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    health_policy = registry.policy_for("GET", "/api/health")
    capability_policy = registry.policy_for("GET", "/api/studio/capabilities")
    detail_policy = registry.policy_for("GET", "/api/studio/capabilities/{capability_id}")
    jobs_list_policy = registry.policy_for("GET", "/api/studio/jobs")
    job_detail_policy = registry.policy_for("GET", "/api/studio/jobs/{job_id}")
    operator_status_policy = registry.policy_for("GET", "/api/studio/operator/status")
    audit_export_policy = registry.policy_for("GET", "/api/studio/audit/export")
    quarantine_export_policy = registry.policy_for(
        "GET",
        "/api/studio/audit/quarantine/export",
    )
    quarantine_archive_policy = registry.policy_for(
        "POST",
        "/api/studio/audit/quarantine/archive",
    )
    quarantine_archive_validate_policy = registry.policy_for(
        "POST",
        "/api/studio/audit/quarantine/archive/validate",
    )
    quarantine_archive_retention_policy = registry.policy_for(
        "GET",
        "/api/studio/audit/quarantine/archive/retention",
    )
    quarantine_archive_restore_policy = registry.policy_for(
        "POST",
        "/api/studio/audit/quarantine/archive/restore",
    )
    quarantine_archive_purge_policy = registry.policy_for(
        "POST",
        "/api/studio/audit/quarantine/archive/purge",
    )
    browser_user_create_policy = registry.policy_for(
        "POST",
        "/api/studio/identity/browser-users",
    )
    artifact_policy = registry.policy_for(
        "GET",
        "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}",
    )

    assert health_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert capability_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert detail_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert jobs_list_policy.visibility is contract["RouteVisibility"].ADMIN
    assert jobs_list_policy.audit_action == "studio.jobs.list"
    assert job_detail_policy.visibility is contract["RouteVisibility"].ADMIN
    assert job_detail_policy.audit_action == "studio.jobs.detail"
    assert operator_status_policy.visibility is contract["RouteVisibility"].ADMIN
    assert audit_export_policy.visibility is contract["RouteVisibility"].ADMIN
    assert quarantine_export_policy.visibility is contract["RouteVisibility"].ADMIN
    assert quarantine_export_policy.audit_action == "studio.audit.quarantine.export"
    assert quarantine_archive_policy.visibility is contract["RouteVisibility"].ADMIN
    assert quarantine_archive_policy.audit_action == "studio.audit.quarantine.archive"
    assert quarantine_archive_validate_policy.visibility is contract["RouteVisibility"].ADMIN
    assert (
        quarantine_archive_validate_policy.audit_action
        == "studio.audit.quarantine.archive.validate"
    )
    assert quarantine_archive_retention_policy.visibility is contract["RouteVisibility"].ADMIN
    assert (
        quarantine_archive_retention_policy.audit_action
        == "studio.audit.quarantine.archive.retention"
    )
    assert quarantine_archive_restore_policy.visibility is contract["RouteVisibility"].ADMIN
    assert (
        quarantine_archive_restore_policy.audit_action == "studio.audit.quarantine.archive.restore"
    )
    assert quarantine_archive_purge_policy.visibility is contract["RouteVisibility"].ADMIN
    assert quarantine_archive_purge_policy.audit_action == "studio.audit.quarantine.archive.purge"
    assert browser_user_create_policy.visibility is contract["RouteVisibility"].ADMIN
    assert browser_user_create_policy.audit_action == "studio.identity.browser_users.create"
    assert artifact_policy.visibility is contract["RouteVisibility"].ADMIN
    assert artifact_policy.audit_action == "studio.jobs.artifact.read"


def test_default_route_policy_registry_reports_unclassified_platform_route() -> None:
    contract = policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    missing = registry.missing_policies(
        (
            ("GET", "/api/health"),
            ("GET", "/api/studio/capabilities"),
            ("POST", "/api/studio/admin"),
        )
    )

    assert missing == ("POST /api/studio/admin",)


def test_route_policy_registry_exports_stable_policy_inventory() -> None:
    contract = policy_contract()
    registry = contract["RoutePolicyRegistry"]()
    health_policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].PUBLIC,
        audit_action="studio.health.read",
    )
    admin_policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].ADMIN,
        audit_action="studio.admin.write",
    )

    registry.register("POST", "/api/studio/admin", admin_policy)
    registry.register("GET", "/api/health", health_policy)

    assert registry.policies() == (
        ("GET", "/api/health", health_policy),
        ("POST", "/api/studio/admin", admin_policy),
    )


def test_studio_app_exposes_route_policy_registry_for_platform_routes() -> None:
    from sc_neurocore.studio.app import create_app  # noqa: PLC0415
    from starlette.routing import Route  # noqa: PLC0415

    app = create_app()
    platform_routes: list[tuple[str, str]] = []
    for route in app.routes:
        if not isinstance(route, Route):
            continue
        if route.path != "/api/health" and not route.path.startswith("/api/studio/"):
            continue
        route_methods = route.methods or set()
        platform_routes.extend(
            (method, route.path) for method in sorted(route_methods) if method != "HEAD"
        )

    missing = app.state.studio_route_policies.missing_policies(tuple(platform_routes))

    assert missing == ()


def test_studio_app_classifies_every_api_and_websocket_route() -> None:
    from sc_neurocore.studio.app import create_app  # noqa: PLC0415
    from starlette.routing import Route, WebSocketRoute  # noqa: PLC0415

    app = create_app()
    route_signatures: list[tuple[str, str]] = []
    for route in app.routes:
        if isinstance(route, Route) and route.path.startswith("/api/"):
            route_methods = route.methods or set()
            route_signatures.extend(
                (method, route.path) for method in sorted(route_methods) if method != "HEAD"
            )
        elif isinstance(route, WebSocketRoute) and route.path.startswith("/ws/"):
            route_signatures.append(("WEBSOCKET", route.path))

    missing = app.state.studio_route_policies.missing_policies(tuple(route_signatures))

    assert missing == ()


def test_default_route_policy_registry_marks_stateful_routes_protected() -> None:
    contract = policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    training_policy = registry.policy_for("POST", "/api/training/start")
    training_checkpoint_export_policy = registry.policy_for(
        "GET",
        "/api/training/checkpoint/{job_id}",
    )
    training_checkpoint_import_policy = registry.policy_for(
        "POST",
        "/api/training/checkpoint/import",
    )
    synth_policy = registry.policy_for("POST", "/api/synth/run")
    synth_terminal_policy = registry.policy_for("POST", "/api/synth/terminal")
    websocket_policy = registry.policy_for("WEBSOCKET", "/ws/progress")
    jobs_status_policy = registry.policy_for("GET", "/api/studio/jobs/status")
    jobs_list_policy = registry.policy_for("GET", "/api/studio/jobs")
    job_detail_policy = registry.policy_for("GET", "/api/studio/jobs/{job_id}")
    artifact_policy = registry.policy_for(
        "GET",
        "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}",
    )

    assert training_policy.visibility is contract["RouteVisibility"].AUTHENTICATED
    assert training_checkpoint_export_policy.visibility is contract["RouteVisibility"].AUTHENTICATED
    assert training_checkpoint_import_policy.visibility is contract["RouteVisibility"].AUTHENTICATED
    assert synth_policy.visibility is contract["RouteVisibility"].ADMIN
    assert synth_terminal_policy.visibility is contract["RouteVisibility"].ADMIN
    assert synth_terminal_policy.audit_action == "studio.synth.terminal"
    assert websocket_policy.visibility is contract["RouteVisibility"].AUTHENTICATED
    assert jobs_status_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert jobs_list_policy.visibility is contract["RouteVisibility"].ADMIN
    assert job_detail_policy.visibility is contract["RouteVisibility"].ADMIN
    assert artifact_policy.visibility is contract["RouteVisibility"].ADMIN
