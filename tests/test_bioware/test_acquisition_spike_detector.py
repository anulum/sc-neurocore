# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeDetector from former test_acquisition.py

"""Focused suite: TestSpikeDetector from former test_acquisition.py."""

from __future__ import annotations

from tests.test_bioware.acquisition_support import *  # noqa: F403

class TestSpikeDetector:
    def test_estimate_noise(self) -> None:
        cfg = MEAConfig(num_channels=10)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        noise = det.estimate_noise(data)
        assert len(noise) == 10
        assert np.all(noise > 0)

    def test_detect_spikes(self) -> None:
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        spikes = det.detect(data)
        assert len(spikes) > 0

    def test_spike_channels(self) -> None:
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        spikes = det.detect(data)
        channels = set(s.channel for s in spikes)
        assert 0 in channels  # We injected spikes on channel 0

    def test_spike_has_timestamp(self) -> None:
        cfg = MEAConfig(num_channels=10)
        det = SpikeDetector(config=cfg)
        spikes = det.detect(_synth_voltage())
        for s in spikes:
            assert s.timestamp_s >= 0

    def test_edge_spike_waveform_is_padded_to_fixed_length(self) -> None:
        # A spike close to the start of the recording yields a truncated raw
        # snippet that must be left-padded to the fixed snippet length.
        cfg = MEAConfig(num_channels=1, sample_rate_hz=20000.0, spike_threshold_sigma=3.0)
        det = SpikeDetector(config=cfg, refractory_samples=0)
        data = np.random.default_rng(0).normal(0.0, 1.0, size=(2000, 1))
        data[5, 0] = -100.0  # strong spike within half a snippet of the edge
        spikes = det.detect(data)
        assert spikes, "edge spike should be detected"
        target_len = 2 * int(2.0 * 20000.0 / 2000.0)
        for spike in spikes:
            assert spike.waveform is not None
            assert len(spike.waveform) == target_len
