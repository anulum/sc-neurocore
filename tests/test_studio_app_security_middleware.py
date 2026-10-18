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
    """Preserve type, referrer and framing headers on ordinary health responses."""
    client = TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    )

    response = client.get("/api/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"


def test_studio_app_generates_request_id_header() -> None:
    """Assign a UUID request identifier when the caller supplies none."""
    client = TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    )

    response = client.get("/api/health")

    request_id = response.headers["x-request-id"]
    assert UUID(request_id).version == 4


def test_studio_app_preserves_valid_inbound_request_id() -> None:
    """Retain a valid caller identifier for end-to-end request correlation."""
    client = TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    )

    response = client.get("/api/health", headers={"x-request-id": "studio-run-42"})

    assert response.headers["x-request-id"] == "studio-run-42"


def test_studio_app_replaces_invalid_inbound_request_id() -> None:
    """Replace an invalid caller identifier with a newly generated UUID."""
    client = TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    )

    response = client.get("/api/health", headers={"x-request-id": "bad request id"})

    request_id = response.headers["x-request-id"]
    assert request_id != "bad request id"
    assert UUID(request_id).version == 4


def test_studio_app_allows_configured_host() -> None:
    """Serve a health request addressed to an explicitly configured host."""
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            allowed_hosts=("studio.example.test",),
        )
    )
    client = TestClient(app, headers={"sec-fetch-site": "none"})

    response = client.get("/api/health", headers={"host": "studio.example.test"})

    assert response.status_code == 200


def test_studio_app_rejects_unconfigured_host() -> None:
    """Refuse a health request addressed outside the configured host set."""
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            allowed_hosts=("studio.example.test",),
        )
    )
    client = TestClient(app, headers={"sec-fetch-site": "none"})

    response = client.get("/api/health", headers={"host": "attacker.example.test"})

    assert response.status_code == 400


def test_studio_app_rejects_oversized_request_body() -> None:
    """Refuse a declared body exceeding the configured byte budget."""
    app = create_app(runtime_settings=StudioRuntimeSettings(max_request_body_bytes=8))
    client = TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"})

    response = client.post("/api/health", content=b"012345678")

    assert response.status_code == 413
    assert response.json()["detail"] == "Studio request body exceeds configured limit."


def test_studio_app_allows_request_body_within_limit() -> None:
    """Let an in-budget body reach ordinary route-method classification."""
    app = create_app(runtime_settings=StudioRuntimeSettings(max_request_body_bytes=8))
    client = TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"})

    response = client.post("/api/health", content=b"01234567")

    assert response.status_code == 405


@pytest.mark.parametrize(("size", "status"), [(8, 405), (9, 413)])
def test_studio_app_counts_streamed_request_bytes(size: int, status: int) -> None:
    """Apply the body budget to streamed uploads without a declared length."""
    client = TestClient(
        create_app(StudioRuntimeSettings(max_request_body_bytes=8)),
        base_url="http://127.0.0.1",
        headers={"sec-fetch-site": "none"},
    )
    response = client.post("/api/health", content=iter([b"0" * size]))

    assert response.status_code == status
    assert response.headers["x-content-type-options"] == "nosniff"
    if status == 413:
        assert response.json()["detail"] == "Studio request body exceeds configured limit."


@pytest.mark.parametrize("enforce", [False, True])
def test_studio_app_cors_preflight_allows_configured_origin(enforce: bool) -> None:
    """Serve allowed preflights before classifying their target's policy."""
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            cors_allowed_origins=("https://studio.example.test",),
            enforce_route_policies=enforce,
        )
    )
    client = TestClient(app, headers={"sec-fetch-site": "none"})

    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://studio.example.test",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://studio.example.test"


@pytest.mark.parametrize("enforce", [False, True])
def test_studio_app_cors_preflight_rejects_unconfigured_origin(enforce: bool) -> None:
    """Keep unconfigured origins outside the CORS allow-list in each mode."""
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            cors_allowed_origins=("https://studio.example.test",),
            enforce_route_policies=enforce,
        )
    )
    client = TestClient(app, headers={"sec-fetch-site": "none"})

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
    """Bind a refused request and its audit event to the caller's identifier."""
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"})

    response = client.post(
        "/api/simulate",
        headers={"x-request-id": "studio-run-42"},
        json={},
    )

    assert response.status_code == 401
    assert response.headers["x-request-id"] == "studio-run-42"
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["request_id"] == "studio-run-42"


@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({"origin": "http://127.0.0.1:8001"}, 200),
        ({"sec-fetch-site": "same-origin"}, 200),
        ({"sec-fetch-site": "none"}, 200),
        ({"origin": "https://other.example.test"}, 403),
        ({"origin": "null", "sec-fetch-site": "none"}, 403),
        ({"sec-fetch-site": "same-site"}, 403),
        ({}, 403),
    ],
)
def test_studio_mutations_require_caller_origin(
    tmp_path: Path, enforce: bool, headers: dict[str, str], status: int
) -> None:
    """Admit only explicit trusted callers before any analysis job is submitted."""
    app = create_app(
        StudioRuntimeSettings(enforce_route_policies=enforce, job_root_path=str(tmp_path / "jobs"))
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/studio/auth/logout",
            headers={
                "x-studio-principal": "operator",
                "x-studio-roles": "studio.viewer",
                **headers,
            },
        )
        assert response.status_code == status
        assert app.state.studio_job_manager.list_records() == ()
        if status == 403:
            refused_job = client.post("/api/analysis/jobs", headers=headers, json={})
            assert refused_job.status_code == 403
            assert app.state.studio_job_manager.list_records() == ()
        assert client.get("/api/health", headers=headers).status_code == 200


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/octet-stream"])
def test_studio_json_route_requires_json_media_type(content_type: str | None) -> None:
    """Refuse missing or non-JSON media types before parsing a JSON login request."""
    headers = {"origin": "http://127.0.0.1:8001"}
    if content_type is not None:
        headers["content-type"] = content_type
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.post("/api/studio/auth/login", headers=headers, content=b"{}")
        assert response.status_code == 415
        assert response.json() == {"detail": "json_content_type_required"}
        handled = client.post(
            "/api/studio/auth/login",
            headers={
                "origin": "http://127.0.0.1:8001",
                "content-type": "application/json; charset=utf-8",
            },
            content=b"{}",
        )
        assert handled.status_code == 422
