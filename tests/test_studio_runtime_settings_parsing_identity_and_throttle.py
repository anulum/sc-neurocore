# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (identity_and_throttle) from former test_studio_runtime_settings_parsing.py

"""Validate identity enforcement and finite browser-login settings before startup."""

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403


def test_studio_runtime_settings_disables_audit_log_by_default() -> None:
    """Keep development auditing unconfigured until a durable path is supplied."""
    settings = build_default_studio_runtime_settings(env={})

    assert settings.audit_log_path is None


def test_studio_runtime_settings_disables_identity_file_by_default() -> None:
    """Keep identity-free development distinct from the enforced identity profile."""
    settings = build_default_studio_runtime_settings(env={})

    assert settings.identity_file_path is None
    assert settings.allow_header_principal is True


def test_studio_runtime_settings_parses_identity_file_and_header_fallback() -> None:
    """Parse an identity path together with disabled preview-principal fallback."""
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_IDENTITY_FILE": "/etc/sc-neurocore/studio-identities.json",
            "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
            "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
        }
    )

    assert settings.identity_file_path == "/etc/sc-neurocore/studio-identities.json"
    assert settings.allow_header_principal is False


@pytest.mark.parametrize(
    ("enforce", "allow_headers", "message"),
    [(False, False, "route policy enforcement"), (True, True, "header principal")],
)
def test_identity_file_requires_consistent_authentication_settings(
    enforce: bool, allow_headers: bool, message: str
) -> None:
    """Refuse an identity store whose route or principal contract is inconsistent."""
    with pytest.raises(ValueError, match=message):
        StudioRuntimeSettings(
            identity_file_path="studio-identities.json",
            enforce_route_policies=enforce,
            allow_header_principal=allow_headers,
        )


def test_studio_runtime_settings_default_browser_login_throttle_is_bounded() -> None:
    """Retain finite default login attempts, failure window and cooldown."""
    settings = build_default_studio_runtime_settings(env={})

    assert settings.browser_login_max_failures == 5
    assert settings.browser_login_failure_window_seconds == 300.0
    assert settings.browser_login_cooldown_seconds == 900.0


def test_studio_runtime_settings_parses_browser_login_throttle() -> None:
    """Preserve explicitly configured browser-login resource budgets."""
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
    """Refuse malformed or non-positive browser-login budget settings."""
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
    """Refuse a preview-principal flag outside the supported boolean syntax."""
    with pytest.raises(ValueError, match="header principal"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "sometimes"}
        )
