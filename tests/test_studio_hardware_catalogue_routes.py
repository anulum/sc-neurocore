# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual application hardware discovery HTTP tests

"""Verify hardware registration discovery through the normal security policy."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.hardware_catalogue import HardwareCatalogue, hardware_catalogue
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.bootstrap import bootstrap_studio_admin_identity


def test_real_app_returns_exact_catalogue_without_creating_a_job(tmp_path: Path) -> None:
    """Public HTTP metadata preserves every source constraint and is read-only."""
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        jobs = client.get("/api/studio/jobs")
        assert jobs.status_code == 200
        response = client.get("/api/hardware/catalogue")
        assert response.status_code == 200
        actual = HardwareCatalogue.model_validate(response.json())
        assert actual == hardware_catalogue()
        assert client.get("/api/studio/jobs").json() == jobs.json()
        assert (
            client.post(
                "/api/hardware/catalogue",
                json={"physical_evidence": ["invented"]},
                headers={"Origin": "http://127.0.0.1:5173"},
            ).status_code
            == 405
        )
        assert (
            HardwareCatalogue.model_validate(client.get("/api/hardware/catalogue").json()) == actual
        )
        assert "HardwareCatalogue" in client.get("/openapi.json").json()["components"]["schemas"]


def test_production_catalogue_is_public_metadata_without_an_identity_claim(tmp_path: Path) -> None:
    """Production policy offers registration metadata without requiring vendor access."""
    identity = tmp_path / "identity.json"
    bootstrap_studio_admin_identity(identity)
    settings = StudioRuntimeSettings(
        deployment_profile="production",
        enforce_route_policies=True,
        allow_header_principal=False,
        identity_file_path=str(identity),
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        response = client.get("/api/hardware/catalogue")
        assert response.status_code == 200, response.text
        catalogue = HardwareCatalogue.model_validate(response.json())
        assert all(not entry.physical_evidence for entry in catalogue.entries)
