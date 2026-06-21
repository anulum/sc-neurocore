# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio EDA runtime setting contracts

"""Runtime-setting tests for Studio EDA process resource ceilings."""

from __future__ import annotations

import pytest

from sc_neurocore.studio.platform import (
    DEFAULT_STUDIO_EDA_PROCESS_CPU_SECONDS,
    DEFAULT_STUDIO_EDA_PROCESS_MEMORY_BYTES,
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)


def test_studio_runtime_settings_default_eda_process_limits_are_bounded() -> None:
    """Default Studio settings keep external EDA child processes bounded."""

    settings = build_default_studio_runtime_settings(env={})

    assert settings.eda_process_cpu_seconds == DEFAULT_STUDIO_EDA_PROCESS_CPU_SECONDS
    assert settings.eda_process_memory_bytes == DEFAULT_STUDIO_EDA_PROCESS_MEMORY_BYTES


def test_studio_runtime_settings_parse_eda_process_limits() -> None:
    """EDA process limits are configurable through environment-style values."""

    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_EDA_PROCESS_CPU_SECONDS": "45.5",
            "SC_NEUROCORE_STUDIO_EDA_PROCESS_MEMORY_BYTES": "268435456",
        }
    )

    assert settings.eda_process_cpu_seconds == 45.5
    assert settings.eda_process_memory_bytes == 268435456


def test_studio_runtime_settings_reject_invalid_eda_process_limits() -> None:
    """Invalid EDA limit settings fail before app startup."""

    with pytest.raises(ValueError, match="EDA process CPU limit"):
        StudioRuntimeSettings(eda_process_cpu_seconds=0)
    with pytest.raises(ValueError, match="EDA process memory limit"):
        StudioRuntimeSettings(eda_process_memory_bytes=0)


def test_studio_runtime_settings_production_requires_eda_ceilings() -> None:
    """Production fails closed when an EDA process ceiling is unbounded."""

    with pytest.raises(ValueError, match="EDA process CPU and memory ceilings"):
        StudioRuntimeSettings(
            deployment_profile="production",
            enforce_route_policies=True,
            allow_header_principal=False,
            identity_file_path="/etc/sc-neurocore/studio-identities.json",
            audit_log_path="/var/log/sc-neurocore/studio-audit.jsonl",
            job_root_path="/var/lib/sc-neurocore/studio-jobs",
            eda_process_cpu_seconds=None,
        )
    with pytest.raises(ValueError, match="EDA process CPU and memory ceilings"):
        StudioRuntimeSettings(
            deployment_profile="production",
            enforce_route_policies=True,
            allow_header_principal=False,
            identity_file_path="/etc/sc-neurocore/studio-identities.json",
            audit_log_path="/var/log/sc-neurocore/studio-audit.jsonl",
            job_root_path="/var/lib/sc-neurocore/studio-jobs",
            eda_process_memory_bytes=None,
        )


def test_studio_runtime_settings_production_accepts_bounded_eda_ceilings() -> None:
    """Production accepts explicit bounded EDA process ceilings."""

    settings = StudioRuntimeSettings(
        deployment_profile="production",
        enforce_route_policies=True,
        allow_header_principal=False,
        identity_file_path="/etc/sc-neurocore/studio-identities.json",
        audit_log_path="/var/log/sc-neurocore/studio-audit.jsonl",
        job_root_path="/var/lib/sc-neurocore/studio-jobs",
        eda_process_cpu_seconds=90.0,
        eda_process_memory_bytes=1_073_741_824,
    )

    assert settings.eda_process_cpu_seconds == 90.0
    assert settings.eda_process_memory_bytes == 1_073_741_824
    with pytest.raises(ValueError, match="EDA process CPU limit"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_EDA_PROCESS_CPU_SECONDS": "not-a-number"}
        )
    with pytest.raises(ValueError, match="EDA process memory limit"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_EDA_PROCESS_MEMORY_BYTES": "not-a-number"}
        )


def test_studio_runtime_settings_parse_sync_analysis_budget() -> None:
    """Synchronous analysis budgets are configurable through environment values."""

    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION": "1000",
            "SC_NEUROCORE_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS": "50000",
            "SC_NEUROCORE_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS": "32",
        }
    )

    assert settings.max_sync_analysis_steps_per_simulation == 1000
    assert settings.max_sync_analysis_total_steps == 50000
    assert settings.max_sync_analysis_simulations == 32


def test_studio_runtime_settings_reject_invalid_sync_analysis_budget() -> None:
    """Invalid synchronous analysis budget settings fail before app startup."""

    with pytest.raises(ValueError, match="steps-per-simulation budget"):
        StudioRuntimeSettings(max_sync_analysis_steps_per_simulation=0)
    with pytest.raises(ValueError, match="total-steps budget"):
        StudioRuntimeSettings(max_sync_analysis_total_steps=0)
    with pytest.raises(ValueError, match="simulation-count budget"):
        StudioRuntimeSettings(max_sync_analysis_simulations=0)
    with pytest.raises(ValueError, match="simulation-count budget must be an integer"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS": "not-a-number"}
        )
