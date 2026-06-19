# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio platform contract tests

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    CapabilityDescriptor,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilityStatus,
    EvidenceClass,
)


def _descriptor(capability_id: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id=capability_id,
        title="Studio Capability Registry",
        summary="Typed capability inventory for Studio feature discovery.",
        status=CapabilityStatus.STABLE,
        requirements=(CapabilityRequirement(name="fastapi", available=True, detail="importable"),),
        evidence=(EvidenceClass.CONTRACT_TEST,),
        ui_placement="Admin",
        docs_path="docs/studio/index.md",
    )


def test_studio_capability_registry_rejects_duplicate_ids() -> None:
    registry = CapabilityRegistry()
    registry.register(_descriptor("studio.capability_registry"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_descriptor("studio.capability_registry"))


def test_studio_capability_health_fails_closed_for_missing_requirement() -> None:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.worker",
            title="Studio Worker",
            summary="Sandboxed Studio worker contract.",
            status=CapabilityStatus.EXPERIMENTAL,
            requirements=(
                CapabilityRequirement(name="worker", available=False, detail="not configured"),
            ),
            evidence=(EvidenceClass.STATIC_INVENTORY,),
            ui_placement="Run",
            docs_path=None,
        )
    )

    health = registry.health("studio.worker")

    assert health.capability_id == "studio.worker"
    assert health.status == CapabilityStatus.UNAVAILABLE
    assert health.healthy is False
    assert health.message == "One or more capability requirements are unavailable."


def test_studio_capabilities_endpoint_returns_safe_inventory() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.get("/api/studio/capabilities")

    assert response.status_code == 200
    payload = response.json()
    capability_ids = {row["capability_id"] for row in payload["capabilities"]}
    assert "studio.capability_registry" in capability_ids
    assert "studio.api" in capability_ids
    assert "secret" not in response.text.lower()
    assert "token" not in response.text.lower()


def test_default_studio_capability_registry_covers_stateful_panels() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.get("/api/studio/capabilities")

    assert response.status_code == 200
    payload = response.json()
    capabilities = {row["capability_id"]: row for row in payload["capabilities"]}
    expected_ids = {
        "studio.simulation_workbench",
        "studio.analysis_suite",
        "studio.compiler_inspector",
        "studio.synthesis_dashboard",
        "studio.training_monitor",
        "studio.network_canvas",
        "studio.project_workspace",
        "studio.export_tools",
    }
    assert expected_ids <= capabilities.keys()
    for capability_id in expected_ids:
        row = capabilities[capability_id]
        assert row["requirements"], capability_id
        assert row["evidence"], capability_id
        assert row["ui_placement"], capability_id
        assert row["docs_path"], capability_id


def test_studio_capability_detail_returns_404_for_unknown_id() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.get("/api/studio/capabilities/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Capability 'missing' not found"
