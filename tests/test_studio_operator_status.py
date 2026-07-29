# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio operator status tests

from __future__ import annotations

import os
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.auth_throttle import StudioLoginThrottleSnapshot
from sc_neurocore.studio.platform import (
    OPERATOR_STATUS_SCHEMA_VERSION,
    AuditSinkStatus,
    CapabilityDescriptor,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilityStatus,
    EvidenceClass,
    RoutePolicy,
    RoutePolicyRegistry,
    RouteVisibility,
    StudioJobManager,
    StudioJobStatusSnapshot,
    StudioRuntimeSettings,
    build_studio_operator_status,
    build_default_studio_route_policy_registry,
)


def _capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.ready",
            title="Ready",
            summary="Ready control-plane capability.",
            status=CapabilityStatus.STABLE,
            requirements=(CapabilityRequirement("ready", True, "available"),),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Admin",
            docs_path="docs/studio/index.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.degraded",
            title="Degraded",
            summary="Degraded control-plane capability.",
            status=CapabilityStatus.DEGRADED,
            requirements=(CapabilityRequirement("degraded", True, "available"),),
            evidence=(EvidenceClass.STATIC_INVENTORY,),
            ui_placement="Admin",
            docs_path=None,
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.missing",
            title="Missing",
            summary="Unavailable control-plane capability.",
            status=CapabilityStatus.EXPERIMENTAL,
            requirements=(CapabilityRequirement("tool", False, "missing"),),
            evidence=(EvidenceClass.STATIC_INVENTORY,),
            ui_placement="Deploy",
            docs_path=None,
        )
    )
    return registry


def _job_status(tmp_path: Path, *, configured: bool) -> StudioJobStatusSnapshot:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=1.0,
        configured=configured,
    )
    return manager.status()


def test_build_studio_operator_status_counts_platform_health(tmp_path: Path) -> None:
    registry = _capability_registry()
    status = build_studio_operator_status(
        settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            identity_file_path="/etc/sc-neurocore/studio-identities.json",
            allow_header_principal=False,
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=7.5,
            job_max_artifact_bytes=4096,
            eda_process_cpu_seconds=12.0,
            eda_process_memory_bytes=268435456,
            browser_login_max_failures=3,
            browser_login_failure_window_seconds=120.0,
            browser_login_cooldown_seconds=900.0,
        ),
        capabilities=tuple(registry.health_all()),
        audit_status=AuditSinkStatus(
            configured=True,
            healthy=True,
            path_configured=True,
            sink_type="jsonl",
        ),
        browser_login_snapshot=StudioLoginThrottleSnapshot(
            active_bucket_count=2,
            locked_bucket_count=1,
            max_retry_after_seconds=58,
        ),
        job_status=_job_status(tmp_path, configured=True),
        route_policy_registry=build_default_studio_route_policy_registry(),
    )

    payload = status.to_public_dict()

    assert payload["schema_version"] == OPERATOR_STATUS_SCHEMA_VERSION
    assert payload["deployment_profile"] == "development"
    assert payload["route_policies"] == {
        "admin_count": 27,
        "authenticated_count": 61,
        "enforced": True,
        "protected_audit_action_count": 88,
        "protected_count": 88,
        "protected_routes_audited": True,
        "public_count": 31,
        "total_count": 119,
    }
    assert payload["identity"] == {
        "configured": True,
        "header_principal_allowed": False,
        "mode": "service_account",
    }
    assert payload["audit"] == {
        "configured": True,
        "healthy": True,
        "integrity_error": None,
        "integrity_verified": None,
        "last_error": None,
        "latest_event_hash": None,
        "path_configured": True,
        "quarantine_reason": None,
        "quarantined_event_count": None,
        "retained_event_count": None,
        "sink_type": "jsonl",
    }
    assert payload["jobs"] == {
        "active_count": 0,
        "allowed_kinds": ["training"],
        "completed_count": 0,
        "configured": True,
        "failed_count": 0,
        "process_count": 0,
        "resource_profiles": [
            {
                "default_timeout_seconds": 1.0,
                "execution_models": ["thread", "process"],
                "kind": "training",
                "max_artifact_bytes": 16777216,
            }
        ],
        "schema_version": "studio.jobs.status.v1",
        "thread_count": 0,
        "timed_out_count": 0,
    }
    assert payload["browser_login"] == {
        "active_bucket_count": 2,
        "cooldown_seconds": 900.0,
        "failure_window_seconds": 120.0,
        "locked_bucket_count": 1,
        "max_retry_after_seconds": 58,
        "max_failures": 3,
    }
    assert payload["resource_limits"] == {
        "eda_process_cpu_seconds": 12.0,
        "eda_process_limits_supported": os.name == "posix",
        "eda_process_memory_bytes": 268435456,
        "job_default_timeout_seconds": 7.5,
        "job_max_artifact_bytes": 4096,
        "max_sync_analysis_simulations": 4096,
        "max_sync_analysis_steps_per_simulation": 5_000_000,
        "max_sync_analysis_total_steps": 200_000_000,
    }
    assert payload["capabilities"] == {
        "degraded_count": 1,
        "experimental_count": 0,
        "healthy_count": 2,
        "stable_count": 1,
        "total_count": 3,
        "unavailable_count": 1,
    }


def test_build_studio_operator_status_reports_header_identity_mode(tmp_path: Path) -> None:
    status = build_studio_operator_status(
        settings=StudioRuntimeSettings(allow_header_principal=True),
        capabilities=(),
        audit_status=AuditSinkStatus(
            configured=False,
            healthy=True,
            path_configured=False,
            sink_type="memory",
        ),
        job_status=_job_status(tmp_path, configured=False),
        route_policy_registry=build_default_studio_route_policy_registry(),
    )

    assert status.identity.to_public_dict() == {
        "configured": False,
        "header_principal_allowed": True,
        "mode": "header_principal",
    }


def test_build_studio_operator_status_reports_disabled_identity_mode(tmp_path: Path) -> None:
    status = build_studio_operator_status(
        settings=StudioRuntimeSettings(allow_header_principal=False),
        capabilities=(),
        audit_status=AuditSinkStatus(
            configured=False,
            healthy=True,
            path_configured=False,
            sink_type="memory",
        ),
        job_status=_job_status(tmp_path, configured=False),
        route_policy_registry=build_default_studio_route_policy_registry(),
    )

    assert status.identity.to_public_dict() == {
        "configured": False,
        "header_principal_allowed": False,
        "mode": "disabled",
    }


def test_build_studio_operator_status_counts_route_policy_inventory(tmp_path: Path) -> None:
    registry = RoutePolicyRegistry()
    registry.register(
        "GET",
        "/api/health",
        RoutePolicy(
            visibility=RouteVisibility.PUBLIC,
            audit_action="studio.health.read",
        ),
    )
    registry.register(
        "GET",
        "/api/studio/session",
        RoutePolicy(
            visibility=RouteVisibility.AUTHENTICATED,
            audit_action="studio.session.read",
        ),
    )
    registry.register(
        "POST",
        "/api/studio/admin",
        RoutePolicy(
            visibility=RouteVisibility.ADMIN,
            audit_action="studio.admin.write",
        ),
    )

    status = build_studio_operator_status(
        settings=StudioRuntimeSettings(enforce_route_policies=True),
        capabilities=(),
        audit_status=AuditSinkStatus(
            configured=False,
            healthy=True,
            path_configured=False,
            sink_type="memory",
        ),
        job_status=_job_status(tmp_path, configured=False),
        route_policy_registry=registry,
    )

    assert status.route_policies.to_public_dict() == {
        "admin_count": 1,
        "authenticated_count": 1,
        "enforced": True,
        "protected_audit_action_count": 2,
        "protected_count": 2,
        "protected_routes_audited": True,
        "public_count": 1,
        "total_count": 3,
    }


def test_operator_status_endpoint_is_admin_protected() -> None:
    app = create_app(StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    denied = client.get("/api/studio/operator/status")
    allowed = client.get(
        "/api/studio/operator/status",
        headers={
            "x-studio-principal": "operator-1",
            "x-studio-roles": "studio.admin",
        },
    )

    assert denied.status_code == 401
    assert denied.json()["detail"] == "missing_principal"
    assert allowed.status_code == 200
    payload = allowed.json()
    assert payload["schema_version"] == OPERATOR_STATUS_SCHEMA_VERSION
    assert payload["deployment_profile"] == "development"
    assert payload["identity"]["mode"] == "header_principal"
    assert payload["route_policies"]["enforced"] is True
    assert payload["route_policies"]["protected_routes_audited"] is True
    assert payload["route_policies"]["protected_count"] > 0
    assert payload["browser_login"] == {
        "active_bucket_count": 0,
        "cooldown_seconds": 900.0,
        "failure_window_seconds": 300.0,
        "locked_bucket_count": 0,
        "max_retry_after_seconds": 0,
        "max_failures": 5,
    }
    assert "token" not in allowed.text.lower()
    assert "/tmp" not in allowed.text
