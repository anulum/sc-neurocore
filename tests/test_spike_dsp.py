# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.spike_dsp

from __future__ import annotations

import numpy as np

from sc_neurocore.spike_dsp import (
    SpikeFIR,
    SpikeIIR,
    spike_convolve,
    spike_fft,
    spike_power_spectrum,
    spike_wavelet_decompose,
)


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


class TestSpikeConvolve:
    def test_basic(self):
        spikes = np.zeros(30, dtype=np.int8)
        spikes[10:15] = 1
        out = spike_convolve(spikes, np.array([0.5, 0.3, 0.2]), threshold=0.3)
        assert out.shape == (30,)


class TestSpikeFFT:
    def test_shape(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random(200) < 0.1).astype(np.int8)
        freqs, mags = spike_fft(spikes, dt=0.001)
        assert len(freqs) == len(mags)
        assert freqs[0] == 0.0

    def test_2d(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 3)) < 0.1).astype(np.int8)
        freqs, mags = spike_fft(spikes, dt=0.001)
        assert mags.shape[1] == 3

    def test_power_spectrum(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random(100) < 0.2).astype(np.int8)
        freqs, psd = spike_power_spectrum(spikes, dt=0.001)
        assert np.all(psd >= 0)


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
