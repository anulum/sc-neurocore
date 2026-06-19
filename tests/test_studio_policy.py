# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy gateway contract tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _policy_contract() -> dict[str, Any]:
    try:
        from sc_neurocore.studio.platform.policy import (  # noqa: PLC0415
            AuditEvent,
            InMemoryAuditSink,
            JsonlAuditSink,
            PolicyGateway,
            Principal,
            RoutePolicyRegistry,
            RoutePolicy,
            RouteVisibility,
            build_default_studio_route_policy_registry,
        )
    except ImportError as exc:
        pytest.fail(f"Studio policy contract is missing: {exc}")
    return {
        "AuditEvent": AuditEvent,
        "InMemoryAuditSink": InMemoryAuditSink,
        "JsonlAuditSink": JsonlAuditSink,
        "PolicyGateway": PolicyGateway,
        "Principal": Principal,
        "RoutePolicyRegistry": RoutePolicyRegistry,
        "RoutePolicy": RoutePolicy,
        "RouteVisibility": RouteVisibility,
        "build_default_studio_route_policy_registry": build_default_studio_route_policy_registry,
    }


def test_policy_gateway_allows_public_route_without_principal() -> None:
    contract = _policy_contract()
    gateway = contract["PolicyGateway"](audit_sink=contract["InMemoryAuditSink"]())
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].PUBLIC,
        audit_action="studio.health.read",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/health")

    assert decision.allowed is True
    assert decision.reason == "public_route"
    assert decision.status_code == 200


def test_policy_gateway_requires_principal_for_authenticated_route() -> None:
    contract = _policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()
    gateway = contract["PolicyGateway"](audit_sink=audit_sink)
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.capabilities.read",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/studio/capabilities")

    assert decision.allowed is False
    assert decision.reason == "missing_principal"
    assert decision.status_code == 401
    assert audit_sink.events[-1].decision == "deny"
    assert audit_sink.events[-1].principal_id is None


def test_policy_gateway_denies_missing_role_and_records_audit_event() -> None:
    contract = _policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()
    gateway = contract["PolicyGateway"](audit_sink=audit_sink)
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        required_roles=frozenset({"studio.admin"}),
        audit_action="studio.policy.write",
    )
    principal = contract["Principal"](principal_id="operator-1", roles=frozenset({"studio.viewer"}))

    decision = gateway.authorize(policy, principal=principal, route="/api/studio/policy")

    assert decision.allowed is False
    assert decision.reason == "missing_required_role"
    assert decision.status_code == 403
    assert audit_sink.events[-1].action == "studio.policy.write"
    assert audit_sink.events[-1].route == "/api/studio/policy"
    assert audit_sink.events[-1].principal_id == "operator-1"
    assert audit_sink.events[-1].decision == "deny"


def test_policy_gateway_allows_authenticated_route_with_required_role() -> None:
    contract = _policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()
    gateway = contract["PolicyGateway"](audit_sink=audit_sink)
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        required_roles=frozenset({"studio.viewer"}),
        audit_action="studio.capabilities.read",
    )
    principal = contract["Principal"](principal_id="operator-3", roles=frozenset({"studio.viewer"}))

    decision = gateway.authorize(policy, principal=principal, route="/api/studio/capabilities")

    assert decision.allowed is True
    assert decision.reason == "authorized"
    assert decision.status_code == 200
    assert audit_sink.events[-1].decision == "allow"
    assert audit_sink.events[-1].principal_id == "operator-3"


def test_jsonl_audit_sink_appends_policy_events(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)

    audit_sink.record(
        contract["AuditEvent"](
            action="studio.simulate.run",
            route="/api/simulate",
            principal_id="operator-7",
            decision="allow",
            reason="authorized",
        )
    )
    audit_sink.record(
        contract["AuditEvent"](
            action="studio.synth.run",
            route="/api/synth/run",
            principal_id=None,
            decision="deny",
            reason="missing_principal",
        )
    )

    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

    assert rows == [
        {
            "action": "studio.simulate.run",
            "decision": "allow",
            "principal_id": "operator-7",
            "reason": "authorized",
            "request_id": None,
            "route": "/api/simulate",
        },
        {
            "action": "studio.synth.run",
            "decision": "deny",
            "principal_id": None,
            "reason": "missing_principal",
            "request_id": None,
            "route": "/api/synth/run",
        },
    ]


def test_jsonl_audit_sink_exposes_configured_path(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)

    assert audit_sink.path == audit_path


def test_policy_gateway_accepts_jsonl_audit_sink(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    gateway = contract["PolicyGateway"](audit_sink=contract["JsonlAuditSink"](audit_path))
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.simulate.run",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/simulate")
    row = json.loads(audit_path.read_text(encoding="utf-8"))

    assert decision.allowed is False
    assert row["decision"] == "deny"
    assert row["reason"] == "missing_principal"


def test_policy_gateway_records_request_id_in_audit_event(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    gateway = contract["PolicyGateway"](audit_sink=contract["JsonlAuditSink"](audit_path))
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.simulate.run",
    )

    decision = gateway.authorize(
        policy,
        principal=None,
        route="/api/simulate",
        request_id="studio-run-42",
    )
    row = json.loads(audit_path.read_text(encoding="utf-8"))

    assert decision.allowed is False
    assert row["request_id"] == "studio-run-42"


def test_policy_gateway_rejects_admin_route_without_admin_role() -> None:
    contract = _policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()
    gateway = contract["PolicyGateway"](audit_sink=audit_sink)
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].ADMIN,
        audit_action="studio.admin.configure",
    )
    principal = contract["Principal"](principal_id="operator-2", roles=frozenset({"studio.viewer"}))

    decision = gateway.authorize(policy, principal=principal, route="/api/studio/admin")

    assert decision.allowed is False
    assert decision.reason == "missing_admin_role"
    assert decision.status_code == 403


def test_route_policy_rejects_empty_audit_action_for_protected_route() -> None:
    contract = _policy_contract()

    with pytest.raises(ValueError, match="audit_action"):
        contract["RoutePolicy"](visibility=contract["RouteVisibility"].AUTHENTICATED)


def test_route_policy_registry_rejects_duplicate_method_path() -> None:
    contract = _policy_contract()
    registry = contract["RoutePolicyRegistry"]()
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].PUBLIC,
        audit_action="studio.health.read",
    )
    registry.register("GET", "/api/health", policy)

    with pytest.raises(ValueError, match="already has a Studio route policy"):
        registry.register("get", "/api/health", policy)


def test_default_route_policy_registry_classifies_platform_routes() -> None:
    contract = _policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    health_policy = registry.policy_for("GET", "/api/health")
    capability_policy = registry.policy_for("GET", "/api/studio/capabilities")
    detail_policy = registry.policy_for("GET", "/api/studio/capabilities/{capability_id}")

    assert health_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert capability_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert detail_policy.visibility is contract["RouteVisibility"].PUBLIC


def test_default_route_policy_registry_reports_unclassified_platform_route() -> None:
    contract = _policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    missing = registry.missing_policies(
        (
            ("GET", "/api/health"),
            ("GET", "/api/studio/capabilities"),
            ("POST", "/api/studio/admin"),
        )
    )

    assert missing == ("POST /api/studio/admin",)


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
        platform_routes.extend((method, route.path) for method in sorted(route_methods) if method != "HEAD")

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
    contract = _policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    training_policy = registry.policy_for("POST", "/api/training/start")
    synth_policy = registry.policy_for("POST", "/api/synth/run")
    websocket_policy = registry.policy_for("WEBSOCKET", "/ws/progress")

    assert training_policy.visibility is contract["RouteVisibility"].AUTHENTICATED
    assert synth_policy.visibility is contract["RouteVisibility"].ADMIN
    assert websocket_policy.visibility is contract["RouteVisibility"].AUTHENTICATED
