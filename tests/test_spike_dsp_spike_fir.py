# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeFIR from former test_spike_dsp.py

"""Focused suite: TestSpikeFIR from former test_spike_dsp.py."""

from __future__ import annotations

from tests.spike_dsp_support import *  # noqa: F403

class TestSpikeFIR:
    def test_basic(self):
        fir = SpikeFIR(coefficients=np.array([0.5, 0.3, 0.2]), threshold=0.5)
        spikes = np.zeros(20, dtype=np.int8)
        spikes[5] = 1
        spikes[6] = 1
        spikes[7] = 1
        out = fir.filter(spikes)
        assert out.shape == (20,)
        assert out.sum() > 0

    def test_2d(self):
        fir = SpikeFIR(coefficients=np.array([0.5, 0.5]), threshold=0.4)
        spikes = np.zeros((20, 3), dtype=np.int8)
        spikes[5:8, 0] = 1
        out = fir.filter(spikes)
        assert out.shape == (20, 3)
