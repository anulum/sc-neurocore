# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for WaveformCodec spike/waveform/background compression

from __future__ import annotations

import struct
import sys
from typing import Any

import numpy as np
import pytest

from sc_neurocore.spike_codec.waveform_codec import WaveformCodec


def _make_waveform(
    T: int = 2000,
    N: int = 16,
    noise_sigma: float = 50.0,
    spike_rate: float = 3.0,
    seed: int = 42,
) -> np.ndarray[Any, Any]:
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
    def test_compress_returns_bytes(self) -> None:
        waveform = _make_waveform()
        codec = WaveformCodec()
        data, result = codec.compress(waveform)
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert result.compression_ratio > 1.0

    def test_detects_spikes(self) -> None:
        waveform = _make_waveform(spike_rate=5.0)
        codec = WaveformCodec(threshold_sigma=4.0)
        _, result = codec.compress(waveform)
        assert result.n_spikes_detected > 0
        assert result.n_templates >= 1

    def test_compression_ratio_reasonable(self) -> None:
        waveform = _make_waveform(T=5000, N=32)
        codec = WaveformCodec(quantize_bits=4)
        _, result = codec.compress(waveform)
        assert result.compression_ratio > 5.0

    def test_spike_timing_lossless(self) -> None:
        waveform = _make_waveform()
        codec = WaveformCodec()
        _, result = codec.compress(waveform)
        assert result.lossless_spikes

    def test_no_spikes_still_compresses(self) -> None:
        """Pure noise (no spikes) should still compress."""
        rng = np.random.RandomState(42)
        noise = rng.randn(2000, 8).astype(np.float32) * 50.0
        codec = WaveformCodec(threshold_sigma=10.0)
        data, result = codec.compress(noise)
        assert result.n_spikes_detected == 0
        assert result.compression_ratio > 1.0

    def test_different_quantization(self) -> None:
        waveform = _make_waveform()
        r4 = WaveformCodec(quantize_bits=4).compress(waveform)[1]
        r8 = WaveformCodec(quantize_bits=8).compress(waveform)[1]
        assert r4.compression_ratio > r8.compression_ratio

    def test_result_fields(self) -> None:
        waveform = _make_waveform()
        _, result = WaveformCodec().compress(waveform)
        assert result.n_channels == 16
        assert result.n_samples == 2000
        assert result.spike_bytes > 0 or result.n_spikes_detected == 0
        assert result.background_bytes > 0

    def test_spike_near_edge_pads_snippet(self) -> None:
        """Spike at end of recording: clip shorter than snippet_samples, triggers padding."""
        rng = np.random.RandomState(42)
        waveform = rng.randn(100, 4).astype(np.float32) * 50.0
        # Inject spike very close to end — snippet will be shorter than 48
        waveform[97:100, 0] = -500.0
        codec = WaveformCodec(threshold_sigma=3.0, snippet_samples=48)
        data, result = codec.compress(waveform)
        assert isinstance(data, bytes)

    def test_template_matching_reuses_template(self) -> None:
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

    def test_very_short_recording_skips_downsample(self) -> None:
        """T < downsample factor (4): background returned without downsampling."""
        rng = np.random.RandomState(42)
        waveform = rng.randn(3, 2).astype(np.float32) * 50.0
        codec = WaveformCodec(threshold_sigma=10.0)
        data, result = codec.compress(waveform)
        assert isinstance(data, bytes)
        assert result.compression_ratio >= 0.1

    def test_empty_background(self) -> None:
        """Empty background array produces empty compressed bytes."""
        codec = WaveformCodec()
        result = codec._compress_background(np.array([]).reshape(0, 0))
        assert result == b""

    def test_constructor_rejects_header_unsafe_parameters(self) -> None:
        """Binary header fields reject values outside the encoded domain."""
        invalid_cases: tuple[tuple[dict[str, Any], str], ...] = (
            ({"threshold_sigma": 0.0}, "threshold_sigma"),
            ({"threshold_sigma": float("nan")}, "threshold_sigma"),
            ({"threshold_sigma": True}, "threshold_sigma"),
            ({"snippet_samples": 0}, "snippet_samples"),
            ({"snippet_samples": 256}, "snippet_samples"),
            ({"snippet_samples": True}, "snippet_samples"),
            ({"max_templates": 0}, "max_templates"),
            ({"max_templates": 65536}, "max_templates"),
            ({"template_threshold": -0.01}, "template_threshold"),
            ({"template_threshold": 1.01}, "template_threshold"),
            ({"template_threshold": False}, "template_threshold"),
            ({"quantize_bits": 0}, "quantize_bits"),
            ({"quantize_bits": 9}, "quantize_bits"),
            ({"quantize_bits": 1.5}, "quantize_bits"),
        )
        for kwargs, field_name in invalid_cases:
            with pytest.raises(ValueError, match=field_name):
                WaveformCodec(**kwargs)

    def test_compress_rejects_non_matrix_waveforms(self) -> None:
        """Raw waveforms must be finite, non-empty time-by-channel matrices."""
        codec = WaveformCodec()
        invalid_inputs = (
            np.arange(8, dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((8, 0), dtype=np.float32),
            np.zeros((2, 2, 2), dtype=np.float32),
        )
        for waveform in invalid_inputs:
            with pytest.raises(ValueError, match="two-dimensional"):
                codec.compress(waveform)

    def test_compress_rejects_nonfinite_waveforms(self) -> None:
        """Invalid ADC values are rejected before spike statistics are derived."""
        waveform = np.zeros((32, 2), dtype=np.float32)
        waveform[3, 1] = np.nan

        with pytest.raises(ValueError, match="finite"):
            WaveformCodec().compress(waveform)

    def test_odd_residual_count_is_padded(self) -> None:
        """Odd residual nibble counts are padded to preserve byte alignment."""
        codec = WaveformCodec(snippet_samples=3)
        templates = [np.array([1.0, -1.0, 0.5], dtype=np.float32)]
        residuals = [np.array([0.25, -0.25, 0.0], dtype=np.float32)]

        encoded = codec._compress_snippets(templates, [0], residuals)

        offset = 2 + 4 + len(templates) * codec.snippet_samples
        offset += 4 + 1
        _, padded_flat_len = struct.unpack("!fI", encoded[offset : offset + 8])
        assert padded_flat_len == 4

    def test_optional_compression_libraries_fall_back_to_zlib(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snippet and background compression work without optional codecs."""
        monkeypatch.setitem(sys.modules, "pywt", None)
        monkeypatch.setitem(sys.modules, "zstandard", None)
        codec = WaveformCodec()
        templates = [np.array([1.0, -1.0, 0.5, -0.25], dtype=np.float32)]
        residuals = [np.array([0.25, -0.25, 0.0, 0.125], dtype=np.float32)]

        snippets = codec._compress_snippets(templates, [0], residuals)
        background = codec._compress_background(np.ones((16, 2), dtype=np.float32))

        assert len(snippets) > 0
        assert len(background) > 12

    def test_compress_rejects_spike_count_header_overflow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dense spike detections cannot silently overflow the two-byte header."""
        codec = WaveformCodec(mode="spike")
        spike_raster = np.zeros((4, 1), dtype=np.int8)
        times_per_channel: list[list[int]] = [[]]

        def fake_detect_spikes(
            waveform: np.ndarray[Any, Any], thresholds: np.ndarray[Any, Any]
        ) -> tuple[np.ndarray[Any, Any], list[list[int]]]:
            return spike_raster, times_per_channel

        def fake_extract_snippets(
            waveform: np.ndarray[Any, Any],
            times_per_ch: list[list[int]],
            n_channels: int,
        ) -> tuple[list[np.ndarray[Any, Any]], list[tuple[int, int]]]:
            return [], [(0, 0)] * 65536

        def fake_template_match(
            snippets: list[np.ndarray[Any, Any]],
        ) -> tuple[list[np.ndarray[Any, Any]], list[int], list[np.ndarray[Any, Any]]]:
            return [], [], []

        monkeypatch.setattr(codec, "_detect_spikes", fake_detect_spikes)
        monkeypatch.setattr(codec, "_extract_snippets", fake_extract_snippets)
        monkeypatch.setattr(codec, "_template_match", fake_template_match)

        with pytest.raises(ValueError, match="header capacity"):
            codec.compress(np.zeros((4, 1), dtype=np.float32))


class TestWaveformCodecMode:
    """Tests for the mode parameter — full / waveform / spike."""

    def test_invalid_mode_raises(self) -> None:
        """Non-existent mode raises ValueError with clear message."""
        with pytest.raises(ValueError, match="mode must be"):
            WaveformCodec(mode="turbo")

    def test_invalid_mode_empty_string(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            WaveformCodec(mode="")

    def test_all_three_modes_compress(self) -> None:
        """All three valid modes produce non-empty compressed bytes."""
        waveform = _make_waveform()
        for mode in ("full", "waveform", "spike"):
            codec = WaveformCodec(mode=mode)
            data, result = codec.compress(waveform)
            assert isinstance(data, bytes), f"mode={mode} did not return bytes"
            assert len(data) > 0, f"mode={mode} produced empty output"
            assert result.compression_ratio > 1.0, f"mode={mode} ratio too low"

    def test_spike_mode_smallest(self) -> None:
        """Spike mode (timing only) produces smallest output."""
        waveform = _make_waveform(T=5000, N=16, spike_rate=5.0)
        data_full, _ = WaveformCodec(mode="full").compress(waveform)
        data_wave, _ = WaveformCodec(mode="waveform").compress(waveform)
        data_spike, _ = WaveformCodec(mode="spike").compress(waveform)
        assert len(data_spike) < len(data_wave), "spike should be smaller than waveform"
        assert len(data_wave) < len(data_full), "waveform should be smaller than full"

    def test_full_mode_has_background_bytes(self) -> None:
        """Full mode includes background LFP bytes in result."""
        waveform = _make_waveform()
        _, result = WaveformCodec(mode="full").compress(waveform)
        assert result.background_bytes > 0

    def test_waveform_mode_no_background(self) -> None:
        """Waveform mode has zero background bytes."""
        waveform = _make_waveform()
        _, result = WaveformCodec(mode="waveform").compress(waveform)
        assert result.background_bytes == 0

    def test_spike_mode_no_background_no_snippets(self) -> None:
        """Spike mode has zero background and zero snippet bytes."""
        waveform = _make_waveform()
        _, result = WaveformCodec(mode="spike").compress(waveform)
        assert result.background_bytes == 0
        assert result.snippet_bytes == 0

    def test_spike_count_same_across_modes(self) -> None:
        """Spike detection is mode-independent — all modes find the same spikes."""
        waveform = _make_waveform(spike_rate=5.0, seed=99)
        results: dict[str, int] = {}
        for mode in ("full", "waveform", "spike"):
            _, r = WaveformCodec(mode=mode, threshold_sigma=4.0).compress(waveform)
            results[mode] = r.n_spikes_detected
        assert results["full"] == results["waveform"] == results["spike"]

    def test_compression_ratio_ordering(self) -> None:
        """Compression ratios: spike > waveform > full."""
        waveform = _make_waveform(T=5000, N=16, spike_rate=5.0)
        ratios: dict[str, float] = {}
        for mode in ("full", "waveform", "spike"):
            _, r = WaveformCodec(mode=mode).compress(waveform)
            ratios[mode] = r.compression_ratio
        assert ratios["spike"] > ratios["waveform"] > ratios["full"]

    def test_mode_byte_in_header(self) -> None:
        """Compressed data header encodes mode as expected byte value."""
        expected_bytes = {"full": 0, "waveform": 1, "spike": 2}
        waveform = _make_waveform()
        for mode, expected in expected_bytes.items():
            data, _ = WaveformCodec(mode=mode).compress(waveform)
            # Header: WFCX(4) + IIHHBBBB = 4+4+4+2+2+1+1+1+1 = 20 bytes
            # mode_byte is last byte of the fixed header (offset 19)
            assert data[:4] == b"WFCX", f"magic mismatch for mode={mode}"
            mode_byte = struct.unpack("B", data[19:20])[0]
            assert mode_byte == expected, f"mode={mode}: expected byte {expected}, got {mode_byte}"

    def test_mode_stored_on_instance(self) -> None:
        """Mode is accessible as an attribute after construction."""
        for mode in ("full", "waveform", "spike"):
            codec = WaveformCodec(mode=mode)
            assert codec.mode == mode

    def test_default_mode_is_full(self) -> None:
        """Default mode without explicit argument is 'full'."""
        codec = WaveformCodec()
        assert codec.mode == "full"

    def test_no_spikes_all_modes(self) -> None:
        """Pure noise (no spikes) compresses in all modes without error."""
        rng = np.random.RandomState(42)
        noise = rng.randn(2000, 8).astype(np.float32) * 50.0
        for mode in ("full", "waveform", "spike"):
            codec = WaveformCodec(mode=mode, threshold_sigma=10.0)
            data, result = codec.compress(noise)
            assert result.n_spikes_detected == 0
            assert isinstance(data, bytes)
