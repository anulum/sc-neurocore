# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio operator status tests

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    OPERATOR_STATUS_SCHEMA_VERSION,
    AuditSinkStatus,
    CapabilityDescriptor,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilityStatus,
    EvidenceClass,
    StudioJobManager,
    StudioJobStatusSnapshot,
    StudioRuntimeSettings,
    build_studio_operator_status,
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
        ),
        capabilities=tuple(registry.health_all()),
        audit_status=AuditSinkStatus(
            configured=True,
            healthy=True,
            path_configured=True,
            sink_type="jsonl",
        ),
        job_status=_job_status(tmp_path, configured=True),
    )

    payload = status.to_public_dict()

    assert payload["schema_version"] == OPERATOR_STATUS_SCHEMA_VERSION
    assert payload["deployment_profile"] == "development"
    assert payload["route_policies"] == {"enforced": True}
    assert payload["identity"] == {
        "configured": True,
        "header_principal_allowed": False,
        "mode": "service_account",
    }
    assert payload["audit"] == {
        "configured": True,
        "healthy": True,
        "last_error": None,
        "path_configured": True,
        "sink_type": "jsonl",
    }
    assert payload["jobs"] == {
        "active_count": 0,
        "allowed_kinds": ["training"],
        "completed_count": 0,
        "configured": True,
        "failed_count": 0,
        "schema_version": "studio.jobs.status.v1",
        "timed_out_count": 0,
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
    )

    assert status.identity.to_public_dict() == {
        "configured": False,
        "header_principal_allowed": False,
        "mode": "disabled",
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
    assert payload["route_policies"] == {"enforced": True}
    assert "token" not in allowed.text.lower()
    assert "/tmp" not in allowed.text
