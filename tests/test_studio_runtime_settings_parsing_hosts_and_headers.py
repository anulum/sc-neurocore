# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (hosts_and_headers) from former test_studio_runtime_settings_parsing.py

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403


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
