# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy authorization gateway tests

"""Authorization gateway tests for Studio policy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from tests.studio_policy_support import UTC, policy_contract


def test_policy_gateway_allows_public_route_without_principal() -> None:
    contract = policy_contract()
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
    contract = policy_contract()
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
    contract = policy_contract()
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
    contract = policy_contract()
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


def test_policy_gateway_accepts_jsonl_audit_sink(tmp_path: Path) -> None:
    contract = policy_contract()
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
    assert row["schema_version"] == "studio.audit.v1"


def test_policy_gateway_records_request_id_in_audit_event(tmp_path: Path) -> None:
    contract = policy_contract()
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


def test_policy_gateway_records_injected_utc_timestamp(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    timestamp = datetime(2026, 6, 19, 3, 52, 0, tzinfo=UTC)
    gateway = contract["PolicyGateway"](
        audit_sink=contract["JsonlAuditSink"](audit_path),
        clock=lambda: timestamp,
    )
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.simulate.run",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/simulate")
    row = json.loads(audit_path.read_text(encoding="utf-8"))

    assert decision.allowed is False
    assert row["timestamp_utc"] == "2026-06-19T03:52:00Z"


def test_policy_gateway_records_default_utc_timestamp(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    gateway = contract["PolicyGateway"](audit_sink=contract["JsonlAuditSink"](audit_path))
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.simulate.run",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/simulate")
    row = json.loads(audit_path.read_text(encoding="utf-8"))

    assert decision.allowed is False
    assert row["timestamp_utc"].endswith("Z")
    assert datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).tzinfo is UTC


def test_policy_gateway_rejects_admin_route_without_admin_role() -> None:
    contract = policy_contract()
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
    contract = policy_contract()

    with pytest.raises(ValueError, match="audit_action"):
        contract["RoutePolicy"](visibility=contract["RouteVisibility"].AUTHENTICATED)
