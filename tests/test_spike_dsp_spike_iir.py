# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeIIR from former test_spike_dsp.py

"""Focused suite: TestSpikeIIR from former test_spike_dsp.py."""

from __future__ import annotations

from tests.spike_dsp_support import *  # noqa: F403

class TestSpikeIIR:
    def test_basic(self):
        iir = SpikeIIR(decay=0.9, threshold=1.0, gain=0.6)
        spikes = np.zeros(30, dtype=np.int8)
        spikes[5:10] = 1
        out = iir.filter(spikes)
        assert out.shape == (30,)
        assert out.sum() > 0

    def test_2d(self):
        iir = SpikeIIR(decay=0.95, threshold=0.5, gain=0.3)
        spikes = np.zeros((20, 4), dtype=np.int8)
        spikes[5:10, :] = 1
        out = iir.filter(spikes)
        assert out.shape == (20, 4)
