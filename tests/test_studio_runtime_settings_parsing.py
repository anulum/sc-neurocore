# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio runtime settings parsing

"""Environment parsing and fail-closed defaults for StudioRuntimeSettings."""

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403

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
