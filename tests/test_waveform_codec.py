# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for WaveformCodec spike/waveform/background compression

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

    def test_spike_near_edge_pads_snippet(self):
        """Spike at end of recording: clip shorter than snippet_samples, triggers padding."""
        rng = np.random.RandomState(42)
        waveform = rng.randn(100, 4).astype(np.float32) * 50.0
        # Inject spike very close to end — snippet will be shorter than 48
        waveform[97:100, 0] = -500.0
        codec = WaveformCodec(threshold_sigma=3.0, snippet_samples=48)
        data, result = codec.compress(waveform)
        assert isinstance(data, bytes)

    def test_template_matching_reuses_template(self):
        """Repeated identical spike shapes trigger template match (best_corr >= threshold)."""
        rng = np.random.RandomState(42)
        waveform = rng.randn(5000, 4).astype(np.float32) * 20.0
        # Same spike template repeated many times per channel
        template = np.zeros(30, dtype=np.float32)
        template[10:15] = -400.0
        template[15:20] = 200.0
        for ch in range(4):
            for i in range(15):
                t = 200 + i * 300
                if t + 30 < 5000:
                    waveform[t : t + 30, ch] += template
        codec = WaveformCodec(threshold_sigma=3.0, max_templates=16, template_threshold=0.7)
        data, result = codec.compress(waveform)
        assert result.n_spikes_detected > 5
        assert result.n_templates >= 1

    def test_very_short_recording_skips_downsample(self):
        """T < downsample factor (4): background returned without downsampling."""
        rng = np.random.RandomState(42)
        waveform = rng.randn(3, 2).astype(np.float32) * 50.0
        codec = WaveformCodec(threshold_sigma=10.0)
        data, result = codec.compress(waveform)
        assert isinstance(data, bytes)
        assert result.compression_ratio >= 0.1

    def test_empty_background(self):
        """Empty background array produces empty compressed bytes."""
        from sc_neurocore.spike_codec.waveform_codec import WaveformCodec

        codec = WaveformCodec()
        result = codec._compress_background(np.array([]).reshape(0, 0))
        assert result == b""


class TestWaveformCodecMode:
    """Tests for the mode parameter — full / waveform / spike."""

    def test_invalid_mode_raises(self):
        """Non-existent mode raises ValueError with clear message."""
        import pytest

        with pytest.raises(ValueError, match="mode must be"):
            WaveformCodec(mode="turbo")

    def test_invalid_mode_empty_string(self):
        import pytest

        with pytest.raises(ValueError, match="mode must be"):
            WaveformCodec(mode="")

    def test_all_three_modes_compress(self):
        """All three valid modes produce non-empty compressed bytes."""
        waveform = _make_waveform()
        for mode in ("full", "waveform", "spike"):
            codec = WaveformCodec(mode=mode)
            data, result = codec.compress(waveform)
            assert isinstance(data, bytes), f"mode={mode} did not return bytes"
            assert len(data) > 0, f"mode={mode} produced empty output"
            assert result.compression_ratio > 1.0, f"mode={mode} ratio too low"

    def test_spike_mode_smallest(self):
        """Spike mode (timing only) produces smallest output."""
        waveform = _make_waveform(T=5000, N=16, spike_rate=5.0)
        data_full, _ = WaveformCodec(mode="full").compress(waveform)
        data_wave, _ = WaveformCodec(mode="waveform").compress(waveform)
        data_spike, _ = WaveformCodec(mode="spike").compress(waveform)
        assert len(data_spike) < len(data_wave), "spike should be smaller than waveform"
        assert len(data_wave) < len(data_full), "waveform should be smaller than full"

    def test_full_mode_has_background_bytes(self):
        """Full mode includes background LFP bytes in result."""
        waveform = _make_waveform()
        _, result = WaveformCodec(mode="full").compress(waveform)
        assert result.background_bytes > 0

    def test_waveform_mode_no_background(self):
        """Waveform mode has zero background bytes."""
        waveform = _make_waveform()
        _, result = WaveformCodec(mode="waveform").compress(waveform)
        assert result.background_bytes == 0

    def test_spike_mode_no_background_no_snippets(self):
        """Spike mode has zero background and zero snippet bytes."""
        waveform = _make_waveform()
        _, result = WaveformCodec(mode="spike").compress(waveform)
        assert result.background_bytes == 0
        assert result.snippet_bytes == 0

    def test_spike_count_same_across_modes(self):
        """Spike detection is mode-independent — all modes find the same spikes."""
        waveform = _make_waveform(spike_rate=5.0, seed=99)
        results = {}
        for mode in ("full", "waveform", "spike"):
            _, r = WaveformCodec(mode=mode, threshold_sigma=4.0).compress(waveform)
            results[mode] = r.n_spikes_detected
        assert results["full"] == results["waveform"] == results["spike"]

    def test_compression_ratio_ordering(self):
        """Compression ratios: spike > waveform > full."""
        waveform = _make_waveform(T=5000, N=16, spike_rate=5.0)
        ratios = {}
        for mode in ("full", "waveform", "spike"):
            _, r = WaveformCodec(mode=mode).compress(waveform)
            ratios[mode] = r.compression_ratio
        assert ratios["spike"] > ratios["waveform"] > ratios["full"]

    def test_mode_byte_in_header(self):
        """Compressed data header encodes mode as expected byte value."""
        import struct

        expected_bytes = {"full": 0, "waveform": 1, "spike": 2}
        waveform = _make_waveform()
        for mode, expected in expected_bytes.items():
            data, _ = WaveformCodec(mode=mode).compress(waveform)
            # Header: WFCX(4) + IIHHBBBB = 4+4+4+2+2+1+1+1+1 = 20 bytes
            # mode_byte is last byte of the fixed header (offset 19)
            assert data[:4] == b"WFCX", f"magic mismatch for mode={mode}"
            mode_byte = struct.unpack("B", data[19:20])[0]
            assert mode_byte == expected, (
                f"mode={mode}: expected byte {expected}, got {mode_byte}"
            )

    def test_mode_stored_on_instance(self):
        """Mode is accessible as an attribute after construction."""
        for mode in ("full", "waveform", "spike"):
            codec = WaveformCodec(mode=mode)
            assert codec.mode == mode

    def test_default_mode_is_full(self):
        """Default mode without explicit argument is 'full'."""
        codec = WaveformCodec()
        assert codec.mode == "full"

    def test_no_spikes_all_modes(self):
        """Pure noise (no spikes) compresses in all modes without error."""
        rng = np.random.RandomState(42)
        noise = rng.randn(2000, 8).astype(np.float32) * 50.0
        for mode in ("full", "waveform", "spike"):
            codec = WaveformCodec(mode=mode, threshold_sigma=10.0)
            data, result = codec.compress(noise)
            assert result.n_spikes_detected == 0
            assert isinstance(data, bytes)
