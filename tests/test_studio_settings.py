# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio runtime settings contract tests

from __future__ import annotations

import json
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)


def _audit_event_hash(row: dict[str, Any]) -> str:
    unsigned_row = dict(row)
    unsigned_row.pop("event_hash", None)
    canonical_row = json.dumps(
        unsigned_row,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_row).hexdigest()


def test_studio_runtime_settings_default_cors_origins_are_loopback_only() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert "http://127.0.0.1:8001" in settings.cors_allowed_origins
    assert "http://localhost:5173" in settings.cors_allowed_origins
    assert "*" not in settings.cors_allowed_origins


def test_studio_runtime_settings_parses_comma_separated_cors_origins() -> None:
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_CORS_ORIGINS": (
                "https://studio.example.test, http://127.0.0.1:9000 "
            )
        }
    )

    assert settings.cors_allowed_origins == (
        "https://studio.example.test",
        "http://127.0.0.1:9000",
    )


def test_studio_runtime_settings_default_websocket_origins_match_cors() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.websocket_allowed_origins == settings.cors_allowed_origins
    assert "*" not in settings.websocket_allowed_origins


def test_studio_runtime_settings_parses_comma_separated_websocket_origins() -> None:
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": (
                "https://studio.example.test, http://127.0.0.1:9000 "
            )
        }
    )

    assert settings.websocket_allowed_origins == (
        "https://studio.example.test",
        "http://127.0.0.1:9000",
    )


def test_studio_runtime_settings_rejects_wildcard_websocket_origin() -> None:
    with pytest.raises(ValueError, match="wildcard WebSocket"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": "http://localhost:5173,*"}
        )


def test_studio_runtime_settings_rejects_empty_websocket_origin_list() -> None:
    with pytest.raises(ValueError, match="WebSocket origins"):
        StudioRuntimeSettings(websocket_allowed_origins=())


def test_studio_runtime_settings_rejects_wildcard_cors_origin() -> None:
    with pytest.raises(ValueError, match="wildcard CORS"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_CORS_ORIGINS": "http://localhost:5173,*"}
        )


def test_studio_runtime_settings_rejects_empty_cors_origin_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        StudioRuntimeSettings(cors_allowed_origins=())


def test_studio_runtime_settings_default_security_headers_are_fail_closed() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.http_security_headers["x-content-type-options"] == "nosniff"
    assert settings.http_security_headers["referrer-policy"] == "no-referrer"
    assert settings.http_security_headers["x-frame-options"] == "DENY"


def test_studio_runtime_settings_default_request_id_header_is_standard() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.request_id_header == "x-request-id"


def test_studio_runtime_settings_disables_route_policy_enforcement_by_default() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.enforce_route_policies is False


def test_studio_runtime_settings_disables_audit_log_by_default() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.audit_log_path is None


def test_studio_runtime_settings_parses_audit_log_path() -> None:
    settings = build_default_studio_runtime_settings(
        env={"SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": "/var/log/sc-neurocore/studio.jsonl"}
    )

    assert settings.audit_log_path == "/var/log/sc-neurocore/studio.jsonl"


def test_studio_runtime_settings_rejects_empty_audit_log_path() -> None:
    with pytest.raises(ValueError, match="audit log path"):
        StudioRuntimeSettings(audit_log_path="")


def test_studio_runtime_settings_parses_route_policy_enforcement_flag() -> None:
    settings = build_default_studio_runtime_settings(
        env={"SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true"}
    )

    assert settings.enforce_route_policies is True


def test_studio_runtime_settings_rejects_invalid_route_policy_enforcement_flag() -> None:
    with pytest.raises(ValueError, match="route policy enforcement"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "sometimes"}
        )


def test_studio_runtime_settings_default_request_body_limit_is_bounded() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.max_request_body_bytes == 1_048_576


def test_studio_runtime_settings_parses_request_body_limit() -> None:
    settings = build_default_studio_runtime_settings(
        env={"SC_NEUROCORE_STUDIO_MAX_REQUEST_BODY_BYTES": "2048"}
    )

    assert settings.max_request_body_bytes == 2048


def test_studio_runtime_settings_rejects_non_positive_request_body_limit() -> None:
    with pytest.raises(ValueError, match="request body limit"):
        StudioRuntimeSettings(max_request_body_bytes=0)


def test_studio_runtime_settings_rejects_invalid_request_body_limit() -> None:
    with pytest.raises(ValueError, match="request body limit"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_MAX_REQUEST_BODY_BYTES": "not-a-number"}
        )


def test_studio_runtime_settings_default_hosts_are_loopback_only() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert "127.0.0.1" in settings.allowed_hosts
    assert "localhost" in settings.allowed_hosts
    assert "*" not in settings.allowed_hosts


def test_studio_runtime_settings_parses_comma_separated_allowed_hosts() -> None:
    settings = build_default_studio_runtime_settings(
        env={"SC_NEUROCORE_STUDIO_ALLOWED_HOSTS": "studio.example.test, 127.0.0.1"}
    )

    assert settings.allowed_hosts == ("studio.example.test", "127.0.0.1")


def test_studio_runtime_settings_rejects_wildcard_allowed_host() -> None:
    with pytest.raises(ValueError, match="wildcard hosts"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_ALLOWED_HOSTS": "localhost,*"}
        )


def test_studio_runtime_settings_rejects_empty_allowed_hosts() -> None:
    with pytest.raises(ValueError, match="allowed hosts"):
        StudioRuntimeSettings(allowed_hosts=())


def test_studio_runtime_settings_rejects_empty_request_id_header() -> None:
    with pytest.raises(ValueError, match="request ID header"):
        StudioRuntimeSettings(request_id_header="")


def test_studio_runtime_settings_rejects_empty_security_header_name() -> None:
    with pytest.raises(ValueError, match="security header names"):
        StudioRuntimeSettings(http_security_headers={"": "nosniff"})


def test_studio_runtime_settings_rejects_empty_security_header_value() -> None:
    with pytest.raises(ValueError, match="security header values"):
        StudioRuntimeSettings(http_security_headers={"x-content-type-options": ""})


def test_studio_app_adds_default_security_headers_to_health_response() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"


def test_studio_app_generates_request_id_header() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    request_id = response.headers["x-request-id"]
    assert UUID(request_id).version == 4


def test_studio_app_preserves_valid_inbound_request_id() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health", headers={"x-request-id": "studio-run-42"})

    assert response.headers["x-request-id"] == "studio-run-42"


def test_studio_app_replaces_invalid_inbound_request_id() -> None:
    client = TestClient(create_app())

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


def test_studio_app_route_policy_enforcement_allows_public_route() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/health")

    assert response.status_code == 200


def test_studio_app_route_policy_enforcement_rejects_missing_principal() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post("/api/simulate", json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "missing_principal"


def test_studio_app_records_policy_events_to_configured_audit_log(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post("/api/simulate", json={})

    assert response.status_code == 401
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["action"] == "studio.simulation.run"
    assert row["decision"] == "deny"
    assert row["principal_id"] is None
    assert row["reason"] == "missing_principal"
    assert row["route"] == "/api/simulate"
    assert row["schema_version"] == "studio.audit.v1"
    assert row["previous_event_hash"] is None
    assert row["event_hash"] == _audit_event_hash(row)
    assert datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).tzinfo is UTC


def test_studio_app_exposes_safe_audit_status(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(runtime_settings=StudioRuntimeSettings(audit_log_path=str(audit_path)))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/studio/audit/status")

    assert response.status_code == 200
    assert response.json() == {
        "configured": True,
        "healthy": True,
        "last_error": None,
        "path_configured": True,
        "sink_type": "jsonl",
    }
    assert str(tmp_path) not in response.text


def test_studio_app_fails_closed_when_policy_audit_append_fails(tmp_path: Path) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(tmp_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/simulate",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
        json={},
    )
    status_response = client.get("/api/studio/audit/status")

    assert response.status_code == 503
    assert response.json()["detail"] == "audit_append_failed"
    assert status_response.status_code == 200
    assert status_response.json()["healthy"] is False
    assert "IsADirectoryError" in status_response.json()["last_error"]
    assert str(tmp_path) not in status_response.text


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
