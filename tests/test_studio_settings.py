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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

UTC = timezone.utc

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
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
    assert settings.deployment_profile == "development"


def test_studio_runtime_settings_accepts_complete_production_profile() -> None:
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "production",
            "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
            "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
            "SC_NEUROCORE_STUDIO_IDENTITY_FILE": "/etc/sc-neurocore/studio-identities.json",
            "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": "/var/log/sc-neurocore/studio-audit.jsonl",
            "SC_NEUROCORE_STUDIO_JOB_ROOT": "/var/lib/sc-neurocore/studio-jobs",
        }
    )

    assert settings.deployment_profile == "production"
    assert settings.enforce_route_policies is True
    assert settings.allow_header_principal is False


@pytest.mark.parametrize(
    ("env_patch", "match"),
    [
        ({"SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "false"}, "route policy"),
        ({"SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "true"}, "header principal"),
        ({"SC_NEUROCORE_STUDIO_IDENTITY_FILE": ""}, "identity file"),
        ({"SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": ""}, "audit log"),
        ({"SC_NEUROCORE_STUDIO_JOB_ROOT": ""}, "job root"),
    ],
)
def test_studio_runtime_settings_rejects_incomplete_production_profile(
    env_patch: dict[str, str],
    match: str,
) -> None:
    env = {
        "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "production",
        "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
        "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
        "SC_NEUROCORE_STUDIO_IDENTITY_FILE": "/etc/sc-neurocore/studio-identities.json",
        "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": "/var/log/sc-neurocore/studio-audit.jsonl",
        "SC_NEUROCORE_STUDIO_JOB_ROOT": "/var/lib/sc-neurocore/studio-jobs",
    }
    env.update(env_patch)

    with pytest.raises(ValueError, match=match):
        build_default_studio_runtime_settings(env=env)


def test_studio_runtime_settings_rejects_unknown_deployment_profile() -> None:
    with pytest.raises(ValueError, match="deployment profile"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "staging"}
        )


def test_studio_runtime_settings_disables_audit_log_by_default() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.audit_log_path is None


def test_studio_runtime_settings_disables_identity_file_by_default() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.identity_file_path is None
    assert settings.allow_header_principal is True


def test_studio_runtime_settings_parses_identity_file_and_header_fallback() -> None:
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_IDENTITY_FILE": "/etc/sc-neurocore/studio-identities.json",
            "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
        }
    )

    assert settings.identity_file_path == "/etc/sc-neurocore/studio-identities.json"
    assert settings.allow_header_principal is False


def test_studio_runtime_settings_default_browser_login_throttle_is_bounded() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.browser_login_max_failures == 5
    assert settings.browser_login_failure_window_seconds == 300.0
    assert settings.browser_login_cooldown_seconds == 900.0


def test_studio_runtime_settings_parses_browser_login_throttle() -> None:
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_COOLDOWN_SECONDS": "120",
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_FAILURE_WINDOW_SECONDS": "30",
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_MAX_FAILURES": "3",
        }
    )

    assert settings.browser_login_max_failures == 3
    assert settings.browser_login_failure_window_seconds == 30.0
    assert settings.browser_login_cooldown_seconds == 120.0


def test_studio_runtime_settings_rejects_invalid_browser_login_throttle() -> None:
    with pytest.raises(ValueError, match="browser login max failures"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_BROWSER_LOGIN_MAX_FAILURES": "not-a-number"}
        )
    with pytest.raises(ValueError, match="browser login failure window"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_BROWSER_LOGIN_FAILURE_WINDOW_SECONDS": "not-a-number"}
        )
    with pytest.raises(ValueError, match="browser login cooldown"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_BROWSER_LOGIN_COOLDOWN_SECONDS": "not-a-number"}
        )
    with pytest.raises(ValueError, match="browser login max failures"):
        StudioRuntimeSettings(browser_login_max_failures=0)
    with pytest.raises(ValueError, match="browser login failure window"):
        StudioRuntimeSettings(browser_login_failure_window_seconds=0.0)
    with pytest.raises(ValueError, match="browser login cooldown"):
        StudioRuntimeSettings(browser_login_cooldown_seconds=0.0)


def test_studio_runtime_settings_rejects_invalid_header_fallback_flag() -> None:
    with pytest.raises(ValueError, match="header principal"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "sometimes"}
        )


def test_studio_runtime_settings_parses_job_root_and_timeout() -> None:
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_JOB_ROOT": "/var/lib/sc-neurocore/studio-jobs",
            "SC_NEUROCORE_STUDIO_JOB_TIMEOUT_SECONDS": "42.5",
            "SC_NEUROCORE_STUDIO_JOB_MAX_ARTIFACT_BYTES": "4096",
        }
    )

    assert settings.job_root_path == "/var/lib/sc-neurocore/studio-jobs"
    assert settings.job_default_timeout_seconds == 42.5
    assert settings.job_max_artifact_bytes == 4096


def test_studio_runtime_settings_default_job_artifact_limit_is_bounded() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.job_max_artifact_bytes == DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES


def test_studio_runtime_settings_rejects_invalid_job_settings() -> None:
    with pytest.raises(ValueError, match="job root path"):
        StudioRuntimeSettings(job_root_path="")
    with pytest.raises(ValueError, match="job timeout"):
        StudioRuntimeSettings(job_default_timeout_seconds=0)
    with pytest.raises(ValueError, match="artifact size"):
        StudioRuntimeSettings(job_max_artifact_bytes=0)
    with pytest.raises(ValueError, match="job timeout"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_JOB_TIMEOUT_SECONDS": "not-a-number"}
        )
    with pytest.raises(ValueError, match="artifact size"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_JOB_MAX_ARTIFACT_BYTES": "not-a-number"}
        )


def test_studio_runtime_settings_parses_audit_log_path() -> None:
    settings = build_default_studio_runtime_settings(
        env={"SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": "/var/log/sc-neurocore/studio.jsonl"}
    )

    assert settings.audit_log_path == "/var/log/sc-neurocore/studio.jsonl"


def test_studio_runtime_settings_disables_audit_rotation_by_default() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.audit_rotation_bytes is None
    assert settings.audit_retained_files == 5


def test_studio_runtime_settings_parses_audit_rotation_policy() -> None:
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_AUDIT_ROTATION_BYTES": "4096",
            "SC_NEUROCORE_STUDIO_AUDIT_RETAINED_FILES": "7",
        }
    )

    assert settings.audit_rotation_bytes == 4096
    assert settings.audit_retained_files == 7


def test_studio_runtime_settings_rejects_empty_audit_log_path() -> None:
    with pytest.raises(ValueError, match="audit log path"):
        StudioRuntimeSettings(audit_log_path="")


def test_studio_runtime_settings_rejects_invalid_audit_rotation_policy() -> None:
    with pytest.raises(ValueError, match="audit rotation"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_AUDIT_ROTATION_BYTES": "not-a-number"}
        )
    with pytest.raises(ValueError, match="retained audit"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_AUDIT_RETAINED_FILES": "not-a-number"}
        )
    with pytest.raises(ValueError, match="audit rotation"):
        StudioRuntimeSettings(audit_rotation_bytes=0)
    with pytest.raises(ValueError, match="retained audit"):
        StudioRuntimeSettings(audit_retained_files=-1)
    with pytest.raises(ValueError, match="retained audit"):
        StudioRuntimeSettings(audit_retained_files=0)


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


def test_studio_app_accepts_bearer_identity_file_principal(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "audit" / "studio.jsonl"
    token_hash = hashlib.sha256(b"admin-token").hexdigest()
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin", "studio.viewer"],
                        "token_sha256": token_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            identity_file_path=str(identity_path),
            allow_header_principal=False,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/export",
        headers={"authorization": "Bearer admin-token"},
    )

    assert response.status_code == 200


def test_studio_app_rejects_invalid_bearer_identity_token(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    token_hash = hashlib.sha256(b"admin-token").hexdigest()
    audit_path = tmp_path / "audit" / "studio.jsonl"
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": token_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            identity_file_path=str(identity_path),
            allow_header_principal=False,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/export",
        headers={"authorization": "Bearer wrong-token"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_identity_token"
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["reason"] == "invalid_identity_token"


def test_studio_app_rejects_header_principal_when_fallback_disabled() -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            allow_header_principal=False,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/simulate",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
        json={},
    )

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
        "integrity_error": None,
        "integrity_verified": True,
        "last_error": None,
        "latest_event_hash": None,
        "path_configured": True,
        "quarantine_reason": None,
        "quarantined_event_count": 0,
        "retained_event_count": 0,
        "sink_type": "jsonl",
    }
    assert str(tmp_path) not in response.text


def test_studio_app_exposes_unhealthy_audit_location_without_path(tmp_path: Path) -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(audit_log_path=str(tmp_path)))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/studio/audit/status")

    assert response.status_code == 200
    assert response.json()["configured"] is True
    assert response.json()["healthy"] is False
    assert response.json()["last_error"] == "AuditPathIsDirectory"
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
    assert status_response.json()["last_error"] == "AuditPathIsDirectory"
    assert str(tmp_path) not in status_response.text


def test_studio_app_exports_audit_events_for_admin_without_paths(tmp_path: Path) -> None:
    from sc_neurocore.studio.platform import AuditEvent, JsonlAuditSink

    audit_path = tmp_path / "audit" / "studio.jsonl"
    JsonlAuditSink(audit_path).record(
        AuditEvent(
            action="studio.simulation.run",
            route="/api/simulate",
            principal_id="operator-1",
            decision="allow",
            reason="authorized",
            request_id="seed-request",
        )
    )
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/export",
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "studio.audit.export.v1"
    assert payload["configured"] is True
    assert payload["integrity_error"] is None
    assert payload["integrity_verified"] is True
    assert payload["latest_event_hash"] == payload["events"][-1]["event_hash"]
    assert payload["quarantine_reason"] is None
    assert payload["quarantined_event_count"] == 0
    assert payload["retained_event_count"] >= 1
    assert payload["sink_type"] == "jsonl"
    assert payload["event_count"] >= 1
    assert payload["events"][0]["action"] == "studio.simulation.run"
    assert str(tmp_path) not in response.text


def test_studio_app_exports_quarantined_audit_events_for_admin_without_paths(
    tmp_path: Path,
) -> None:
    from sc_neurocore.studio.platform import AuditEvent, JsonlAuditSink

    audit_path = tmp_path / "audit" / "studio.jsonl"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text('{"schema_version":"studio.audit.v1"}\n', encoding="utf-8")
    JsonlAuditSink(audit_path).record(
        AuditEvent(
            action="studio.simulation.run",
            route="/api/simulate",
            principal_id="operator-1",
            decision="allow",
            reason="authorized",
            request_id="seed-request",
        )
    )
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/quarantine/export",
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "studio.audit.quarantine.export.v1"
    assert payload["configured"] is True
    assert payload["event_count"] == 1
    assert payload["events"][0]["quarantine_reason"] == "legacy_or_unverifiable_rows"
    assert payload["quarantine_reason"] == "legacy_or_unverifiable_rows"
    assert payload["retained_event_count"] >= 2
    assert payload["sink_type"] == "jsonl"
    assert str(tmp_path) not in response.text


def test_studio_app_rejects_audit_export_without_admin_role(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/export",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "missing_admin_role"


def test_studio_app_rejects_quarantine_export_without_admin_role(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/quarantine/export",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "missing_admin_role"


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
