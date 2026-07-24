# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (route_policy_and_body) from former test_studio_runtime_settings_parsing.py

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403

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
