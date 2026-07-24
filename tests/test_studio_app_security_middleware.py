# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio app security middleware

"""HTTP host, body-limit, CORS preflight, request-id, and security-header contracts."""

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403


def test_studio_app_adds_default_security_headers_to_health_response() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.get("/api/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"


def test_studio_app_generates_request_id_header() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.get("/api/health")

    request_id = response.headers["x-request-id"]
    assert UUID(request_id).version == 4


def test_studio_app_preserves_valid_inbound_request_id() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.get("/api/health", headers={"x-request-id": "studio-run-42"})

    assert response.headers["x-request-id"] == "studio-run-42"


def test_studio_app_replaces_invalid_inbound_request_id() -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.get("/api/health", headers={"x-request-id": "bad request id"})

    request_id = response.headers["x-request-id"]
    assert request_id != "bad request id"
    assert UUID(request_id).version == 4


def test_studio_app_allows_configured_host() -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            allowed_hosts=("studio.example.test",),
        )
    )
    client = TestClient(app)

    response = client.get("/api/health", headers={"host": "studio.example.test"})

    assert response.status_code == 200


def test_studio_app_rejects_unconfigured_host() -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            allowed_hosts=("studio.example.test",),
        )
    )
    client = TestClient(app)

    response = client.get("/api/health", headers={"host": "attacker.example.test"})

    assert response.status_code == 400


def test_studio_app_rejects_oversized_request_body() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(max_request_body_bytes=8))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post("/api/health", content=b"012345678")

    assert response.status_code == 413
    assert response.json()["detail"] == "Studio request body exceeds configured limit."


def test_studio_app_allows_request_body_within_limit() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(max_request_body_bytes=8))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post("/api/health", content=b"01234567")

    assert response.status_code == 405


def test_studio_app_cors_preflight_allows_configured_origin() -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            cors_allowed_origins=("https://studio.example.test",)
        )
    )
    client = TestClient(app)

    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://studio.example.test",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://studio.example.test"


def test_studio_app_cors_preflight_rejects_unconfigured_origin() -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            cors_allowed_origins=("https://studio.example.test",)
        )
    )
    client = TestClient(app)

    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://attacker.example.test",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_studio_app_correlates_policy_audit_with_request_id(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/simulate",
        headers={"x-request-id": "studio-run-42"},
        json={},
    )

    assert response.status_code == 401
    assert response.headers["x-request-id"] == "studio-run-42"
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["request_id"] == "studio-run-42"
