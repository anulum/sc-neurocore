# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCochleaAdapter from former test_sensor_fusion.py

"""Focused suite: TestCochleaAdapter from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403

class TestCochleaAdapter:
    def test_freq_to_channel_boundaries(self):
        coch = CochleaAdapter(num_channels=64)
        assert coch.freq_to_channel(10.0) == 0
        assert coch.freq_to_channel(25000.0) == 63

    def test_freq_to_channel_mid(self):
        coch = CochleaAdapter(num_channels=64)
        ch = coch.freq_to_channel(1000.0)
        assert 0 < ch < 63

    def test_log_scale_ordering(self):
        coch = CochleaAdapter(num_channels=64)
        ch_low = coch.freq_to_channel(100.0)
        ch_mid = coch.freq_to_channel(1000.0)
        ch_high = coch.freq_to_channel(10000.0)
        assert ch_low < ch_mid < ch_high

    def test_encode_spikes(self):
        ts = np.arange(5, dtype=np.float64) * 100
        freqs = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
        coch = CochleaAdapter(num_channels=32)
        stream = coch.encode_spikes(ts, freqs)
        assert stream.modality == SensorModality.COCHLEA
        assert stream.num_events == 5
