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
    with pytest.raises(ValueError, match="EDA process CPU limit"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_EDA_PROCESS_CPU_SECONDS": "not-a-number"}
        )
    with pytest.raises(ValueError, match="EDA process memory limit"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_EDA_PROCESS_MEMORY_BYTES": "not-a-number"}
        )
