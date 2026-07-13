# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware analysis tests

"""Tests for culture, LFP, burst, and latency analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.bioware.bioware import (
    CultureHealth,
    DetectedSpike,
    LatencyBudget,
    LFPBand,
    detect_network_bursts,
    extract_lfp_power,
)


def _synth_voltage(
    n_samples: int,
    n_channels: int,
    seed: int = 42,
) -> np.ndarray[Any, Any]:
    """Generate deterministic finite voltage data for spectral tests."""
    return np.random.default_rng(seed).normal(0.0, 5.0, size=(n_samples, n_channels))


class TestCultureHealth:
    def test_healthy_culture(self) -> None:
        ch = CultureHealth(min_active_channels=3)
        counts = np.array([10, 15, 20, 5, 8, 0, 0, 0])
        result = ch.assess(counts, duration_s=1.0)
        assert result["is_viable"] is True

    def test_dead_culture(self) -> None:
        ch = CultureHealth(min_active_channels=5)
        counts = np.zeros(60)
        result = ch.assess(counts, duration_s=1.0)
        assert result["health_score"] == 0.0
        assert result["is_viable"] is False

    def test_bursting_detection(self) -> None:
        ch = CultureHealth(burst_threshold_hz=50.0)
        counts = np.array([100, 200, 5, 3])
        result = ch.assess(counts, duration_s=1.0)
        assert result["bursting_channels"] == 2

    def test_excessive_firing_rate_caps_health(self) -> None:
        # A mean rate above the hyperactivity ceiling scales the health score
        # down rather than leaving it at 1.0.
        ch = CultureHealth(min_active_channels=1, max_firing_rate_hz=10.0)
        counts = np.full(8, 1000.0)
        result = ch.assess(counts, duration_s=1.0)
        assert result["health_score"] < 1.0


# ── BioHybridSession Tests ───────────────────────────────────────────


class TestLFPExtraction:
    def test_default_bands(self) -> None:
        data = _synth_voltage(n_samples=2000, n_channels=5)
        result = extract_lfp_power(data, sample_rate_hz=20000.0)
        assert "delta" in result
        assert "gamma" in result
        assert result["delta"].shape == (5,)

    def test_custom_band(self) -> None:
        data = _synth_voltage(n_samples=2000, n_channels=5)
        bands = [LFPBand("custom", 10.0, 50.0)]
        result = extract_lfp_power(data, sample_rate_hz=20000.0, bands=bands)
        assert "custom" in result
        assert np.all(result["custom"] >= 0)


# ── Latency Budget Tests (Gap 3) ───────────────────────────────────────


class TestLatencyBudget:
    def test_within_budget(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        assert lb.record(500.0) is True
        assert lb.violations == 0

    def test_exceeds_budget(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        assert lb.record(1500.0) is False
        assert lb.violations == 1

    def test_compliance_ratio(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        lb.record(500.0)
        lb.record(500.0)
        lb.record(1500.0)
        assert lb.compliance_ratio == pytest.approx(2.0 / 3.0)

    def test_p99_latency(self) -> None:
        lb = LatencyBudget()
        for i in range(100):
            lb.record(float(i))
        assert lb.p99_latency_us > 90.0

    def test_mean_latency(self) -> None:
        lb = LatencyBudget()
        lb.record(100.0)
        lb.record(300.0)
        assert lb.mean_latency_us == pytest.approx(200.0)

    def test_compliance_ratio_empty_history(self) -> None:
        assert LatencyBudget().compliance_ratio == 1.0


# ── PharmModel Tests (Gap 4) ───────────────────────────────────────────


class TestNetworkBurstDetection:
    def test_synchronised_burst(self) -> None:
        rng = np.random.default_rng(42)
        spikes: list[DetectedSpike] = []
        # Background: sparse spikes
        for i in range(100):
            spikes.append(
                DetectedSpike(
                    channel=int(rng.integers(0, 10)),
                    timestamp_s=float(rng.uniform(0, 1)),
                    amplitude_uv=-30,
                )
            )
        # Burst: many spikes from many channels at t=0.5
        for ch in range(8):
            for _ in range(10):
                spikes.append(
                    DetectedSpike(
                        channel=ch,
                        timestamp_s=0.5 + float(rng.uniform(-0.005, 0.005)),
                        amplitude_uv=-50,
                    )
                )
        bursts = detect_network_bursts(
            spikes, bin_width_s=0.01, threshold_sigma=2.0, min_channels=5
        )
        assert len(bursts) > 0
        assert bursts[0].participating_channels >= 5

    def test_no_burst(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=float(i), amplitude_uv=-30) for i in range(10)
        ]
        bursts = detect_network_bursts(spikes, min_channels=3)
        assert len(bursts) == 0

    def test_empty_spikes(self) -> None:
        assert detect_network_bursts([]) == []

    def test_same_timestamp_spikes_have_no_temporal_span(self) -> None:
        spikes = [
            DetectedSpike(channel=ch, timestamp_s=0.25, amplitude_uv=-40.0) for ch in range(4)
        ]

        assert detect_network_bursts(spikes, min_channels=1) == []

    def test_uniform_bin_counts_have_no_burst_threshold(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.00, amplitude_uv=-40.0),
            DetectedSpike(channel=1, timestamp_s=0.01, amplitude_uv=-40.0),
            DetectedSpike(channel=2, timestamp_s=0.02, amplitude_uv=-40.0),
        ]

        assert detect_network_bursts(spikes, bin_width_s=0.01, min_channels=1) == []


# ── Artifact Rejection Tests (Gap 7) ───────────────────────────────────
