# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeFFT from former test_spike_dsp.py

"""Focused suite: TestSpikeFFT from former test_spike_dsp.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.spike_dsp import (
    spike_fft,
    spike_power_spectrum,
)


class TestSpikeFFT:
    """The FFT and power spectrum of spike trains."""

    def test_shape(self) -> None:
        """The FFT returns matching frequencies and magnitudes, starting at 0 Hz."""
        rng = np.random.RandomState(42)
        spikes = (rng.random(200) < 0.1).astype(np.int8)
        freqs, mags = spike_fft(spikes, dt=0.001)
        assert len(freqs) == len(mags)
        assert freqs[0] == 0.0

    def test_2d(self) -> None:
        """A two-dimensional input gives one magnitude column per channel."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 3)) < 0.1).astype(np.int8)
        freqs, mags = spike_fft(spikes, dt=0.001)
        assert mags.shape[1] == 3

    def test_power_spectrum(self) -> None:
        """The power spectrum is non-negative."""
        rng = np.random.RandomState(42)
        spikes = (rng.random(100) < 0.2).astype(np.int8)
        freqs, psd = spike_power_spectrum(spikes, dt=0.001)
        assert np.all(psd >= 0)
