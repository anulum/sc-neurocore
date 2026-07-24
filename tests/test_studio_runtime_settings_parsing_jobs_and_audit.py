# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (jobs_and_audit) from former test_studio_runtime_settings_parsing.py

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403

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
