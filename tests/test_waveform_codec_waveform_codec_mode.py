# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveformCodecMode from former test_waveform_codec.py

"""Focused suite: TestWaveformCodecMode from former test_waveform_codec.py."""

from __future__ import annotations

from tests.waveform_codec_support import *  # noqa: F403

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
