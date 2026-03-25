# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.spike_codec.waveform_codec

from __future__ import annotations

import numpy as np

from sc_neurocore.spike_codec.waveform_codec import WaveformCodec


def _make_waveform(T=2000, N=16, noise_sigma=50.0, spike_rate=3.0, seed=42):
    """Generate synthetic raw electrode waveform with spikes."""
    rng = np.random.RandomState(seed)
    waveform = rng.randn(T, N).astype(np.float32) * noise_sigma
    template = np.zeros(48)
    template[15:20] = -200
    template[20:25] = 100
    template[25:30] = -30
    for ch in range(N):
        n_spikes = max(1, int(spike_rate * T / 20000))
        times = rng.choice(range(100, T - 100), size=min(n_spikes, T - 200), replace=False)
        for t in times:
            s, e = max(0, t - 24), min(T, t + 24)
            waveform[s:e, ch] += template[: e - s]
    return waveform


class TestWaveformCodec:
    def test_compress_returns_bytes(self):
        waveform = _make_waveform()
        codec = WaveformCodec()
        data, result = codec.compress(waveform)
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert result.compression_ratio > 1.0

    def test_detects_spikes(self):
        waveform = _make_waveform(spike_rate=5.0)
        codec = WaveformCodec(threshold_sigma=4.0)
        _, result = codec.compress(waveform)
        assert result.n_spikes_detected > 0
        assert result.n_templates >= 1

    def test_compression_ratio_reasonable(self):
        waveform = _make_waveform(T=5000, N=32)
        codec = WaveformCodec(quantize_bits=4)
        _, result = codec.compress(waveform)
        assert result.compression_ratio > 5.0

    def test_spike_timing_lossless(self):
        waveform = _make_waveform()
        codec = WaveformCodec()
        _, result = codec.compress(waveform)
        assert result.lossless_spikes

    def test_no_spikes_still_compresses(self):
        """Pure noise (no spikes) should still compress."""
        rng = np.random.RandomState(42)
        noise = rng.randn(2000, 8).astype(np.float32) * 50.0
        codec = WaveformCodec(threshold_sigma=10.0)
        data, result = codec.compress(noise)
        assert result.n_spikes_detected == 0
        assert result.compression_ratio > 1.0

    def test_different_quantization(self):
        waveform = _make_waveform()
        r4 = WaveformCodec(quantize_bits=4).compress(waveform)[1]
        r8 = WaveformCodec(quantize_bits=8).compress(waveform)[1]
        assert r4.compression_ratio > r8.compression_ratio

    def test_result_fields(self):
        waveform = _make_waveform()
        _, result = WaveformCodec().compress(waveform)
        assert result.n_channels == 16
        assert result.n_samples == 2000
        assert result.spike_bytes > 0 or result.n_spikes_detected == 0
        assert result.background_bytes > 0
