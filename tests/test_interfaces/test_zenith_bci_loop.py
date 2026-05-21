# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Zenith BCI loop tests

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.interfaces import ZenithBCILoop, ZenithBCILoopConfig


def _waveform(samples: int = 96, channels: int = 4) -> np.ndarray:
    data = np.zeros((samples, channels), dtype=np.float32)
    data[12, 0] = -25.0
    data[40, 1] = -30.0
    data[72, 3] = -35.0
    return data


def test_zenith_loop_produces_budget_trace_and_command() -> None:
    loop = ZenithBCILoop(
        ZenithBCILoopConfig(
            n_channels=4,
            sampling_rate_hz=30_000,
            gpu_lanes=4,
            latency_budget_ms=10.0,
            threshold_sigma=4.0,
            snippet_samples=16,
        )
    )

    result = loop.process_stream(_waveform(), window_start_us=100, pathway_name="neuropixels")

    assert result.command in (0, 1)
    assert result.spike_count == 3
    assert result.feedback_active_channels == 3
    assert result.pathway_name == "neuropixels"
    assert set(result.latency_breakdown_ms) == {"ingest", "codec", "decode", "feedback"}
    assert result.total_latency_ms < 10.0
    assert result.latency_budget_met is True
    assert result.to_dict()["schema_version"] == "sc-neurocore.zenith-bci-loop.v1"


def test_zenith_loop_is_deterministic_for_same_input() -> None:
    loop = ZenithBCILoop(ZenithBCILoopConfig(n_channels=4, gpu_lanes=2))
    waveform = _waveform()

    first = loop.process_stream(waveform, window_start_us=123)
    second = loop.process_stream(waveform, window_start_us=123)

    assert first.to_dict() == second.to_dict()


def test_zenith_loop_flags_budget_violation_with_tight_budget() -> None:
    loop = ZenithBCILoop(
        ZenithBCILoopConfig(
            n_channels=64,
            sampling_rate_hz=1_000,
            gpu_lanes=1,
            latency_budget_ms=0.01,
        )
    )
    waveform = np.zeros((512, 64), dtype=np.float32)

    result = loop.process_stream(waveform)

    assert result.total_latency_ms > result.latency_budget_ms
    assert result.latency_budget_met is False


def test_zenith_loop_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="gpu_lanes"):
        ZenithBCILoopConfig(n_channels=4, gpu_lanes=0)

    loop = ZenithBCILoop(ZenithBCILoopConfig(n_channels=4))
    with pytest.raises(ValueError, match="pathway_name"):
        loop.process_stream(_waveform(), pathway_name="")
