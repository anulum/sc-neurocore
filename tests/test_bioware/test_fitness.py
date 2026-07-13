# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware fitness-adapter tests

"""Tests for MEA-derived evolutionary fitness metrics."""

from __future__ import annotations

from typing import Any, cast

import pytest

from sc_neurocore.bioware.bioware import DetectedSpike, mea_fitness_hook


class TestMEAFitnessHook:
    """``mea_fitness_hook`` converts MEA spike dynamics into the
    ``{"accuracy", "energy_mw", "latency_ms"}`` triple consumed by the
    evo_substrate ``ReplicationEngine`` fitness function.
    """

    def test_empty_spikes_returns_floor(self) -> None:
        r = mea_fitness_hook([])
        assert r == {"accuracy": 0.1, "energy_mw": 0.0, "latency_ms": 0.0}

    def test_empty_spikes_preserve_explicit_measured_latency(self) -> None:
        result = mea_fitness_hook([], measured_latency_ms=3.75)
        assert result["latency_ms"] == pytest.approx(3.75)

    def test_near_target_rate_scores_high(self) -> None:
        # 10 spikes on a single channel, target_rate=10 → mean_rate = 10 → accuracy 0.99 ceiling.
        spikes = [
            DetectedSpike(channel=0, timestamp_s=i * 0.01, amplitude_uv=-40.0) for i in range(10)
        ]
        r = mea_fitness_hook(spikes, target_rate=10.0)
        assert r["accuracy"] == pytest.approx(0.99, abs=1e-9)

    def test_off_target_rate_penalised(self) -> None:
        # 100 spikes on one channel, target 10 → rate_error ratio = 9 → accuracy floor.
        spikes = [
            DetectedSpike(channel=0, timestamp_s=i * 0.001, amplitude_uv=-40.0) for i in range(100)
        ]
        r = mea_fitness_hook(spikes, target_rate=10.0)
        assert r["accuracy"] == pytest.approx(0.1, abs=1e-9)

    def test_energy_scales_with_spike_count(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0)] * 20
        r = mea_fitness_hook(spikes)
        assert r["energy_mw"] == pytest.approx(20 * 0.5)

    def test_duration_converts_counts_to_rates(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=i * 0.05, amplitude_uv=-40.0) for i in range(20)
        ]
        r = mea_fitness_hook(spikes, target_rate=10.0, duration_s=2.0)
        assert r["accuracy"] == pytest.approx(0.99, abs=1e-9)

    def test_target_rate_zero_returns_floor(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0)]
        r = mea_fitness_hook(spikes, target_rate=0.0)
        assert r["accuracy"] == pytest.approx(0.1, abs=1e-9)

    def test_channel_key_used_not_channel_id(self) -> None:
        # Regression guard: previous implementation accessed ``s.channel_id``
        # which doesn't exist on DetectedSpike and raised AttributeError on
        # any non-empty input.
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0),
            DetectedSpike(channel=1, timestamp_s=0.0, amplitude_uv=-40.0),
        ]
        r = mea_fitness_hook(spikes)
        assert {"accuracy", "energy_mw", "latency_ms"} == set(r.keys())

    def test_latency_uses_measured_closed_loop_value(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.25, amplitude_uv=-40.0)]
        r = mea_fitness_hook(spikes, measured_latency_ms=3.75)
        assert r["latency_ms"] == pytest.approx(3.75)

    def test_latency_uses_first_response_after_stimulus(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.090, amplitude_uv=-40.0),
            DetectedSpike(channel=0, timestamp_s=0.125, amplitude_uv=-40.0),
            DetectedSpike(channel=1, timestamp_s=0.140, amplitude_uv=-40.0),
        ]
        r = mea_fitness_hook(spikes, stimulus_time_s=0.100)
        assert r["latency_ms"] == pytest.approx(25.0)

    def test_latency_zero_when_no_spike_follows_stimulus(self) -> None:
        # Every spike precedes the stimulus, so there is no causal response.
        spikes = [DetectedSpike(channel=0, timestamp_s=0.05, amplitude_uv=-40.0)]
        r = mea_fitness_hook(spikes, stimulus_time_s=0.1)
        assert r["latency_ms"] == 0.0

    def test_latency_non_finite_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timestamp_s must be finite"):
            DetectedSpike(channel=0, timestamp_s=float("inf"), amplitude_uv=-40.0)

    def test_response_latency_empty_spikes_without_measured_is_zero(self) -> None:
        from sc_neurocore.bioware.bioware import _mea_response_latency_ms

        assert _mea_response_latency_ms([], stimulus_time_s=None, measured_latency_ms=None) == 0.0

    def test_latency_without_stimulus_uses_first_spike_timestamp(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.006, amplitude_uv=-40.0),
            DetectedSpike(channel=1, timestamp_s=0.014, amplitude_uv=-40.0),
        ]
        r = mea_fitness_hook(spikes)
        assert r["latency_ms"] == pytest.approx(6.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"measured_latency_ms": -1.0},
            {"stimulus_time_s": float("nan")},
            {"duration_s": 0.0},
        ],
    )
    def test_rejects_invalid_fitness_timing_parameters(self, kwargs: dict[str, float]) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0)]
        with pytest.raises(ValueError):
            mea_fitness_hook(spikes, **cast(Any, kwargs))
