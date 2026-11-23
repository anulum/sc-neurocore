# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCochleaAdapter from former test_sensor_fusion.py

"""Focused suite: TestCochleaAdapter from former test_sensor_fusion.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.fusion.sensor_fusion import (
    CochleaAdapter,
    SensorModality,
)


class TestCochleaAdapter:
    """Frequency mapping and spike encoding of the cochlea adapter."""

    def test_freq_to_channel_boundaries(self) -> None:
        """Frequencies below and above the band map to the first and last channels."""
        coch = CochleaAdapter(num_channels=64)
        assert coch.freq_to_channel(10.0) == 0
        assert coch.freq_to_channel(25000.0) == 63

    def test_freq_to_channel_mid(self) -> None:
        """A mid-band frequency maps to an inner channel."""
        coch = CochleaAdapter(num_channels=64)
        ch = coch.freq_to_channel(1000.0)
        assert 0 < ch < 63

    def test_log_scale_ordering(self) -> None:
        """Higher frequencies map to higher channels."""
        coch = CochleaAdapter(num_channels=64)
        ch_low = coch.freq_to_channel(100.0)
        ch_mid = coch.freq_to_channel(1000.0)
        ch_high = coch.freq_to_channel(10000.0)
        assert ch_low < ch_mid < ch_high

    def test_encode_spikes(self) -> None:
        """Five tones encode as five cochlea events."""
        ts = np.arange(5, dtype=np.float64) * 100
        freqs = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
        coch = CochleaAdapter(num_channels=32)
        stream = coch.encode_spikes(ts, freqs)
        assert stream.modality == SensorModality.COCHLEA
        assert stream.num_events == 5
