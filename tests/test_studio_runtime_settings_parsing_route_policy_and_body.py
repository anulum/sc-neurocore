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


def _job_root(app: Any) -> Path:
    return Path(app.state.studio_job_manager.root)


def test_an_unconfigured_job_root_is_private_to_its_process() -> None:
    """Durability is for a configured root; an unconfigured one is scratch.

    One fixed path under the system temp directory was shared by every Studio
    on the host: a second process — or a later run — opened the first one's
    ledger and reported its jobs as its own, and on a multi-user machine the
    directory belonged to whichever user created it first.
    """

    first = _job_root(create_app())
    second = _job_root(create_app())

    assert first != second
    assert first.name.startswith("sc-neurocore-studio-jobs-")
    assert first.is_dir()
    # Private, not world-readable: it holds a ledger and job artifacts.
    assert first.stat().st_mode & 0o077 == 0


def test_a_configured_job_root_is_used_as_given(tmp_path: Path) -> None:
    configured = tmp_path / "jobs"

    app = create_app(StudioRuntimeSettings(job_root_path=str(configured)))

    assert _job_root(app) == configured
