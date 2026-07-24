# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (route_policy) from former test_studio_app_policy_and_audit.py

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403


def test_studio_app_route_policy_enforcement_allows_public_route() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/health")

    assert response.status_code == 200


def test_studio_app_route_policy_enforcement_rejects_unclassified_route() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/unclassified")

    assert response.status_code == 403
    assert response.json()["detail"] == "unclassified_route"


def test_studio_app_route_policy_enforcement_rejects_missing_principal() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post("/api/simulate", json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "missing_principal"


def test_studio_app_route_policy_enforcement_allows_authenticated_principal() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/simulate",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
        json={},
    )

    assert response.status_code != 401


def test_studio_app_route_policy_enforcement_rejects_missing_admin_role() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/synth/run",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
        json={"verilog": "module top; endmodule", "target": "ice40"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "missing_admin_role"
