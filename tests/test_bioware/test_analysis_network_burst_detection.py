# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetworkBurstDetection from former test_analysis.py

"""Focused suite: TestNetworkBurstDetection from former test_analysis.py."""

from __future__ import annotations

from tests.test_bioware.analysis_support import *  # noqa: F403


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
