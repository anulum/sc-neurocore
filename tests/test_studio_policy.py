# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy gateway contract tests

from __future__ import annotations

from typing import Any

import pytest


def _policy_contract() -> dict[str, Any]:
    try:
        from sc_neurocore.studio.platform.policy import (  # noqa: PLC0415
            InMemoryAuditSink,
            PolicyGateway,
            Principal,
            RoutePolicy,
            RouteVisibility,
        )
    except ImportError as exc:
        pytest.fail(f"Studio policy contract is missing: {exc}")
    return {
        "InMemoryAuditSink": InMemoryAuditSink,
        "PolicyGateway": PolicyGateway,
        "Principal": Principal,
        "RoutePolicy": RoutePolicy,
        "RouteVisibility": RouteVisibility,
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
