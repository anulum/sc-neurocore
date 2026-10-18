# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (security_and_profile) from former test_studio_runtime_settings_parsing.py

"""Validate deployment requirements and listener binding before the API starts."""

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403


def test_studio_runtime_settings_default_security_headers_are_fail_closed() -> None:
    """Require the default response headers before any profile-specific overrides."""
    settings = build_default_studio_runtime_settings(env={})

    assert settings.http_security_headers["x-content-type-options"] == "nosniff"
    assert settings.http_security_headers["referrer-policy"] == "no-referrer"
    assert settings.http_security_headers["x-frame-options"] == "DENY"


def test_studio_runtime_settings_default_request_id_header_is_standard() -> None:
    """Use the conventional request correlation header when no override is supplied."""
    settings = build_default_studio_runtime_settings(env={})

    assert settings.request_id_header == "x-request-id"


def test_studio_runtime_settings_disables_route_policy_enforcement_by_default() -> None:
    """Keep the explicitly unprotected development profile distinct from lab settings."""
    settings = build_default_studio_runtime_settings(env={})

    assert settings.enforce_route_policies is False
    assert settings.deployment_profile == "development"


def test_studio_runtime_settings_accepts_complete_production_profile() -> None:
    """Accept production settings only with identities, enforcement and persistent state."""
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
    """Refuse each missing production requirement at settings construction."""
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
    """Reject an undeclared deployment profile instead of selecting development."""
    with pytest.raises(ValueError, match="deployment profile"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "staging"}
        )


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.20", "studio.example.test"])
def test_development_listener_refuses_non_loopback_bind(host: str) -> None:
    """Refuse a network listener before a development application can start."""
    with pytest.raises(ValueError, match="non-loopback bind requires the production profile"):
        StudioRuntimeSettings().validate_bind_host(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_development_listener_accepts_loopback_bind(host: str) -> None:
    """Retain local IPv4, IPv6 and localhost launch compatibility."""
    StudioRuntimeSettings().validate_bind_host(host)
