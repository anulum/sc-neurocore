# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveformCodec from former test_waveform_codec.py

"""Focused suite: TestWaveformCodec from former test_waveform_codec.py."""

from __future__ import annotations

from tests.waveform_codec_support import *  # noqa: F403


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
