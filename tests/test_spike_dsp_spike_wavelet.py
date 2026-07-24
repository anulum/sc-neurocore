# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeWavelet from former test_spike_dsp.py

"""Focused suite: TestSpikeWavelet from former test_spike_dsp.py."""

from __future__ import annotations

from tests.spike_dsp_support import *  # noqa: F403


class TestSpikeWavelet:
    def test_basic(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random(100) < 0.15).astype(np.int8)
        scales = spike_wavelet_decompose(spikes, n_scales=3)
        assert len(scales) == 3
        for s in scales:
            assert s.shape == (100,)

    def test_2d(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 4)) < 0.15).astype(np.int8)
        scales = spike_wavelet_decompose(spikes, n_scales=3)
        assert len(scales) == 3
        for s in scales:
            assert s.shape == (100, 4)
