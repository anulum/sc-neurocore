# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — WaveformCodec decoder and lossless mode

"""Round trips of ``WaveformCodec.decompress`` for every mode, and refusals of malformed streams."""

from __future__ import annotations

import struct
import sys
import zlib
from typing import Any

import numpy as np
import pytest

from sc_neurocore.spike_codec import waveform_codec as wc
from sc_neurocore.spike_codec.codec import SpikeCodec
from sc_neurocore.spike_codec.waveform_codec import WaveformCodec
from tests.waveform_codec_support import _make_waveform


def _lossless_round_trip(samples: np.ndarray[Any, Any], mode: str = "lossless") -> bytes:
    """Compress losslessly, check the exact round trip and return the stream."""
    data, result = WaveformCodec(mode=mode).compress(samples)
    decoded = WaveformCodec.decompress(data)
    assert decoded.mode == mode
    assert decoded.lossless_waveform
    assert decoded.spike_raster is None
    assert decoded.waveform is not None
    assert decoded.waveform.dtype == np.int16
    np.testing.assert_array_equal(decoded.waveform, samples)
    assert result.lossless_waveform and not result.lossless_spikes
    assert result.compressed_bytes == len(data)
    return data


class TestLosslessMode:
    """The lossless mode returns every input sample exactly."""

    def test_random_full_range_multichannel(self) -> None:
        """Uniform int16 noise over three channels, including the range ends."""
        rng = np.random.default_rng(1)
        samples = rng.integers(-32768, 32768, size=(1500, 3)).astype(np.int16)
        samples[0, 0], samples[1, 0] = -32768, 32767
        _lossless_round_trip(samples)

    def test_sparse_value_table_compresses(self) -> None:
        """A 10-bit ADC scaled by a non-integer factor uses the value table and compresses."""
        t = np.arange(20000)
        levels = np.round(300 * np.sin(t / 50.0)).astype(np.int64)
        scaled = np.round(levels * 64.0626)
        assert np.abs(scaled).max() < 32767
        samples = scaled.astype(np.int16).reshape(-1, 1)
        data = _lossless_round_trip(samples, "lossless_rice")
        header = 4 + struct.calcsize("!IIHHBBBB") + 4 + 4
        assert data[header] == 1, "value-table mapping expected"
        assert samples.nbytes / len(data) > 4.0

    def test_affine_lattice_is_used_for_dense_values(self) -> None:
        """Dense values on a lattice of step 4 use the affine mapping."""
        rng = np.random.default_rng(2)
        samples = (rng.integers(-1000, 1000, size=(5000, 1)) * 4 + 3).astype(np.int16)
        data = _lossless_round_trip(samples, "lossless_rice")
        header = 4 + struct.calcsize("!IIHHBBBB") + 4 + 4
        assert data[header] == 0, "affine mapping expected"
        offset, step = struct.unpack("!iI", data[header + 1 : header + 9])
        assert step == 4 and offset == int(samples.min())

    def test_escaped_residuals(self) -> None:
        """Full-scale alternation forces escaped (verbatim) residuals inside a quiet block."""
        samples = np.zeros((600, 1), dtype=np.int16)
        samples[300::2, 0] = 32767
        samples[301::2, 0] = -32768
        _lossless_round_trip(samples)

    def test_constant_short_and_single_sample(self) -> None:
        """Constant channels, two samples and one sample decode exactly."""
        _lossless_round_trip(np.full((9, 2), -32768, dtype=np.int16))
        _lossless_round_trip(np.array([[7], [-7]], dtype=np.int16))
        _lossless_round_trip(np.array([[5]], dtype=np.int16))

    def test_higher_order_predictor_on_smooth_signal(self) -> None:
        """A quadratic ramp is coded with a predictor of order above 1."""
        t = np.arange(4000, dtype=np.int64)
        samples = ((t * t) // 1000 - 8000).astype(np.int16).reshape(-1, 1)
        data = _lossless_round_trip(samples)
        assert samples.nbytes / len(data) > 8.0

    def test_float_integer_input_is_accepted(self) -> None:
        """Integer-valued float arrays are accepted and decoded as int16."""
        samples = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
        data, _ = WaveformCodec(mode="lossless").compress(samples)
        np.testing.assert_array_equal(
            WaveformCodec.decompress(data).waveform, samples.astype(np.int16)
        )

    @pytest.mark.parametrize(
        ("samples", "message"),
        [
            (np.array([[0.5]]), "integer samples"),
            (np.array([[40000.0]]), "int16 range"),
            (np.array([[-40000.0]]), "int16 range"),
            (np.array([[np.nan]]), "finite"),
            (np.zeros(3), "two-dimensional"),
        ],
    )
    def test_refuses_input_it_cannot_return_exactly(
        self, samples: np.ndarray[Any, Any], message: str
    ) -> None:
        """Non-integer, out-of-range, non-finite and malformed input is refused."""
        with pytest.raises(ValueError, match=message):
            WaveformCodec(mode="lossless").compress(samples)


class TestLossyModesDecode:
    """Spike raster exact; waveform and background reconstructions bounded."""

    def test_spike_mode_returns_the_detected_raster(self) -> None:
        """The decoded raster has exactly the detected spikes and no waveform."""
        waveform = _make_waveform(T=4000, N=4, spike_rate=60.0)
        data, result = WaveformCodec(mode="spike").compress(waveform)
        decoded = WaveformCodec.decompress(data)
        assert (
            decoded.mode == "spike" and decoded.waveform is None and not decoded.lossless_waveform
        )
        assert decoded.spike_raster is not None
        assert int(decoded.spike_raster.sum()) == result.n_spikes_detected > 0

    def test_waveform_mode_reconstructs_snippets(self) -> None:
        """Snippet regions are reconstructed close to the input; elsewhere zero."""
        waveform = _make_waveform(T=4000, N=2, noise_sigma=5.0, spike_rate=60.0)
        codec = WaveformCodec(mode="waveform", max_templates=4)
        data, result = codec.compress(waveform)
        decoded = WaveformCodec.decompress(data)
        assert decoded.waveform is not None and decoded.spike_raster is not None
        assert result.n_spikes_detected > 0
        half = codec.snippet_samples // 2
        for ch in range(waveform.shape[1]):
            for t in np.flatnonzero(decoded.spike_raster[:, ch]):
                start, end = max(0, t - half), min(waveform.shape[0], t + half)
                original = waveform[start:end, ch]
                error = np.abs(decoded.waveform[start:end, ch] - original).max()
                assert error < 0.25 * np.abs(original).max()
        outside = np.ones(waveform.shape, dtype=bool)
        for ch in range(waveform.shape[1]):
            for t in np.flatnonzero(decoded.spike_raster[:, ch]):
                outside[max(0, t - half) : t + half, ch] = False
        assert np.all(decoded.waveform[outside] == 0)

    def test_scaled_spikes_keep_their_amplitude(self) -> None:
        """Spikes of one shape and different amplitudes decode at their own amplitude."""
        rng = np.random.default_rng(3)
        waveform = rng.normal(0.0, 1.0, size=(2000, 1)).astype(np.float32)
        shape = np.zeros(48, dtype=np.float32)
        shape[15:20], shape[20:25], shape[25:30] = -200.0, 100.0, -30.0
        for t, amplitude in ((300, 1.0), (900, 2.0), (1500, 0.5)):
            waveform[t - 24 : t + 24, 0] += amplitude * shape
        data, result = WaveformCodec(mode="waveform", max_templates=4).compress(waveform)
        assert result.n_templates == 1 and result.n_spikes_detected == 3
        decoded = WaveformCodec.decompress(data)
        assert decoded.waveform is not None
        for t, amplitude in ((300, 1.0), (900, 2.0), (1500, 0.5)):
            region = slice(t - 20, t + 20)
            error = np.abs(decoded.waveform[region, 0] - waveform[region, 0]).max()
            assert error < 0.1 * 200.0 * amplitude, (t, error)

    def test_full_mode_background_has_level_and_bounded_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slow offset signal without spikes keeps its level within the quantisation step.

        PyWavelets is made unavailable so the bound tests the closed-loop quantiser alone; the
        optional wavelet denoising is a deliberate further loss.
        """
        monkeypatch.setitem(sys.modules, "pywt", None)
        t = np.arange(3200, dtype=np.float32)
        background = (1500.0 + 200.0 * np.sin(t / 400.0)).astype(np.float32)
        waveform = np.stack([background, background - 300.0], axis=1)
        codec = WaveformCodec(mode="full", threshold_sigma=50.0, quantize_bits=8)
        data, result = codec.compress(waveform)
        assert result.n_spikes_detected == 0
        decoded = WaveformCodec.decompress(data)
        assert decoded.waveform is not None
        rows = waveform.shape[0] // wc.WAVEFORM_CODEC_BACKGROUND_DOWNSAMPLE
        means = waveform[: rows * 16].reshape(rows, 16, 2).mean(axis=1)
        reconstructed = decoded.waveform[::16][:rows]
        step = np.abs(np.diff(means, axis=0)).max() / 127
        assert np.abs(reconstructed - means).max() < 3 * step + 1e-3

    def test_full_mode_short_signal_without_downsampling(self) -> None:
        """Fewer samples than the downsampling factor keep their own rows."""
        waveform = np.array([[100.0], [101.0], [99.0], [100.5]], dtype=np.float32)
        decoded = WaveformCodec.decompress(
            WaveformCodec(mode="full", threshold_sigma=50.0).compress(waveform)[0]
        )
        assert decoded.waveform is not None
        assert np.abs(decoded.waveform - waveform).max() < 1.0

    def test_full_mode_with_spikes_on_two_channels(self) -> None:
        """Spikes and background together decode with bounded error."""
        waveform = _make_waveform(T=3200, N=2, noise_sigma=5.0, spike_rate=60.0)
        data, result = WaveformCodec(mode="full", max_templates=4).compress(waveform)
        decoded = WaveformCodec.decompress(data)
        assert decoded.waveform is not None and result.n_spikes_detected > 0
        rms_error = float(np.sqrt(np.mean((decoded.waveform - waveform) ** 2)))
        assert rms_error < float(np.sqrt(np.mean(waveform**2)))

    def test_zlib_fallback_streams_decode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Streams written without zstandard (zlib sections) decode the same way."""
        monkeypatch.setitem(sys.modules, "zstandard", None)
        monkeypatch.setitem(sys.modules, "pywt", None)
        waveform = _make_waveform(T=1600, N=2, noise_sigma=5.0, spike_rate=60.0)
        data, _ = WaveformCodec(mode="full", max_templates=2).compress(waveform)
        decoded = WaveformCodec.decompress(data)
        assert decoded.waveform is not None and decoded.waveform.shape == waveform.shape

    def test_inflate_recognises_both_formats(self) -> None:
        """``_inflate`` reads zlib streams and zstd frames by their magic."""
        payload = b"abc" * 50
        assert wc._inflate(zlib.compress(payload)) == payload
        zstd = pytest.importorskip("zstandard")
        assert wc._inflate(zstd.ZstdCompressor().compress(payload)) == payload


def _body(data: bytes) -> bytes:
    """Return a WFC2 stream without its CRC-32 trailer."""
    return data[:-4]


def _reseal(body: bytes) -> bytes:
    """Seal a (deliberately inconsistent) body with a valid CRC-32, as a faulty encoder would."""
    return wc._seal(body)


class TestMalformedStreams:
    """Corrupted, inconsistent or legacy streams are refused with a reason, never decoded silently."""

    def _spike_stream(self) -> bytes:
        return WaveformCodec(mode="spike").compress(_make_waveform(T=2000, N=2, spike_rate=60.0))[0]

    def test_legacy_stream_spike_raster_only(self) -> None:
        """A WFCX stream (no checksum) decodes its spike raster; its waveform is refused."""
        legacy_spike = b"WFCX" + _body(self._spike_stream())[4:]
        assert WaveformCodec.decompress(legacy_spike).spike_raster is not None
        full = WaveformCodec(mode="full").compress(_make_waveform(T=2000, N=2))[0]
        with pytest.raises(ValueError, match="legacy WFCX"):
            WaveformCodec.decompress(b"WFCX" + _body(full)[4:])
        for mode in ("lossless", "lossless_rice"):
            lossless = WaveformCodec(mode=mode).compress(np.zeros((4, 1), dtype=np.int16))[0]
            with pytest.raises(ValueError, match="only in the WFC2"):
                WaveformCodec.decompress(b"WFCX" + _body(lossless)[4:])

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda d: d[:10], "shorter than the WaveformCodec header"),
            (lambda d: d[:22], "header and checksum"),
            (lambda d: b"NOPE" + d[4:], "not a WaveformCodec"),
            (lambda d: d[:-1], "checksum mismatch"),
            (lambda d: d + b"x", "checksum mismatch"),
            (lambda d: d[:30] + bytes([d[30] ^ 1]) + d[31:], "checksum mismatch"),
        ],
    )
    def test_corrupted_bytes_are_refused(self, mutate: Any, message: str) -> None:
        """Truncation, a foreign magic, extra bytes and a single flipped bit are refused."""
        with pytest.raises(ValueError, match=message):
            WaveformCodec.decompress(mutate(self._spike_stream()))

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda b: b[:19] + bytes([9]) + b[20:], "unknown WaveformCodec mode"),
            (lambda b: b[:4] + struct.pack("!I", 0) + b[8:], "at least one each"),
            (lambda b: b[:-1], "truncated"),
            (lambda b: b[:21], "truncated"),
            (lambda b: b + b"x", "trailing bytes"),
        ],
    )
    def test_inconsistent_sealed_streams_are_refused(self, mutate: Any, message: str) -> None:
        """A checksummed stream with a bad mode, dimension or section length is still refused."""
        with pytest.raises(ValueError, match=message):
            WaveformCodec.decompress(_reseal(mutate(_body(self._spike_stream()))))

    def test_declared_size_limit(self) -> None:
        """A stream declaring more samples than ``max_samples`` is refused before allocation."""
        data = self._spike_stream()
        assert WaveformCodec.decompress(data, max_samples=4000).spike_raster is not None
        with pytest.raises(ValueError, match="max_samples=3999"):
            WaveformCodec.decompress(data, max_samples=3999)
        huge = _reseal(_body(data)[:8] + struct.pack("!I", 5_898_242) + _body(data)[12:])
        with pytest.raises(ValueError, match="max_samples"):
            WaveformCodec.decompress(huge)

    def test_snippet_and_background_inconsistencies(self) -> None:
        """Counts that disagree with the header, and a missing background, are refused."""
        waveform = _make_waveform(T=2000, N=2, spike_rate=60.0)
        body = _body(WaveformCodec(mode="waveform").compress(waveform)[0])
        n_templates = struct.unpack("!H", body[12:14])[0]
        with pytest.raises(ValueError, match="template count"):
            WaveformCodec.decompress(
                _reseal(body[:12] + struct.pack("!H", n_templates + 1) + body[14:])
            )
        n_snippets = struct.unpack("!H", body[14:16])[0]
        with pytest.raises(ValueError, match="snippet count"):
            WaveformCodec.decompress(
                _reseal(body[:14] + struct.pack("!H", n_snippets + 1) + body[16:])
            )
        with pytest.raises(ValueError, match="background section"):
            WaveformCodec.decompress(_reseal(body[:19] + bytes([0]) + body[20:]))

    def test_snippet_section_id_count_must_match_header(self) -> None:
        """A snippet section listing a different number of template IDs than the header is refused."""
        codec = WaveformCodec(mode="waveform")
        body = _body(codec.compress(_make_waveform(T=2000, N=2, spike_rate=60.0))[0])
        header = 4 + struct.calcsize(wc._HEADER_FORMAT)
        (spike_length,) = struct.unpack("!I", body[header : header + 4])
        n_templates = struct.unpack("!H", body[12:14])[0]
        ids_at = header + 4 + spike_length + 4 + 2 + 4 + n_templates * codec.snippet_samples
        (n_ids,) = struct.unpack("!I", body[ids_at : ids_at + 4])
        changed = body[:ids_at] + struct.pack("!I", n_ids + 1) + body[ids_at + 4 :]
        with pytest.raises(ValueError, match="snippet count disagrees with the header"):
            WaveformCodec.decompress(_reseal(changed))

    def test_short_section_parse_errors_become_value_errors(self) -> None:
        """A snippet section too short to parse raises ValueError, not struct.error."""
        body = _body(
            WaveformCodec(mode="waveform").compress(_make_waveform(T=2000, N=2, spike_rate=60.0))[0]
        )
        header = 4 + struct.calcsize(wc._HEADER_FORMAT)
        (spike_length,) = struct.unpack("!I", body[header : header + 4])
        snippets_at = header + 4 + spike_length
        truncated = (
            body[:snippets_at] + struct.pack("!I", 3) + b"\x00\x01\x02" + struct.pack("!I", 0)
        )
        with pytest.raises(ValueError, match="malformed WaveformCodec stream"):
            WaveformCodec.decompress(_reseal(truncated))

    def test_snippet_placement_must_match_raster(self) -> None:
        """A spike section with fewer spikes than the stream's snippets is refused."""
        waveform = _make_waveform(T=2000, N=2, spike_rate=60.0)
        many = _body(WaveformCodec(mode="waveform").compress(waveform)[0])
        few = _body(WaveformCodec(mode="waveform", threshold_sigma=8.0).compress(waveform)[0])
        header = 4 + struct.calcsize(wc._HEADER_FORMAT)

        def spike_section(data: bytes) -> tuple[bytes, int]:
            (length,) = struct.unpack("!I", data[header : header + 4])
            return data[header : header + 4 + length], header + 4 + length

        few_section, _ = spike_section(few)
        _, rest = spike_section(many)
        assert struct.unpack("!H", few[14:16])[0] < struct.unpack("!H", many[14:16])[0]
        spliced = many[:header] + few_section + many[rest:]
        with pytest.raises(ValueError, match="does not match the decoded spike raster"):
            WaveformCodec.decompress(_reseal(spliced))

    def test_corrupted_compressed_sections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A damaged zlib or zstd section, or a zstd frame without zstandard, is a ValueError."""
        with pytest.raises(ValueError, match="corrupted"):
            wc._inflate(b"\x78\x9c" + b"\x00" * 8)
        with pytest.raises(ValueError, match="corrupted"):
            wc._inflate(wc._ZSTD_MAGIC + b"\x00" * 8)
        monkeypatch.setitem(sys.modules, "zstandard", None)
        with pytest.raises(ValueError, match="needs the zstandard package"):
            wc._inflate(wc._ZSTD_MAGIC + b"\x00" * 8)

    def _lossless_channel(self, samples: np.ndarray[Any, Any]) -> tuple[bytes, int]:
        body = _body(WaveformCodec(mode="lossless_rice").compress(samples)[0])
        start = 4 + struct.calcsize("!IIHHBBBB") + 4 + 4
        return body, start

    @pytest.mark.parametrize("backend", ["python", "rust"])
    def test_lossless_channel_corruption(self, backend: str) -> None:
        """Mapping kind, table length, predictor order and Rice lengths are checked on both backends."""
        samples = np.round(np.arange(600) % 37 * 64.06).astype(np.int16).reshape(-1, 1)
        body, start = self._lossless_channel(samples)
        assert body[start] == 1
        with pytest.raises(ValueError, match="level mapping"):
            WaveformCodec.decompress(
                backend=backend, data=_reseal(body[:start] + bytes([7]) + body[start + 1 :])
            )
        n_table = struct.unpack("!I", body[start + 1 : start + 5])[0]
        with pytest.raises(ValueError, match="value table length"):
            WaveformCodec.decompress(
                backend=backend,
                data=_reseal(body[: start + 1] + struct.pack("!I", 10**6) + body[start + 5 :]),
            )
        with pytest.raises(ValueError, match="value table length"):
            WaveformCodec.decompress(
                backend=backend,
                data=_reseal(body[: start + 1] + struct.pack("!I", 0) + body[start + 5 :]),
            )
        pos = start + 5
        for _ in range(n_table):
            _, pos = SpikeCodec._decode_varint(body, pos)
        with pytest.raises(ValueError, match="predictor order"):
            WaveformCodec.decompress(
                backend=backend, data=_reseal(body[:pos] + bytes([9]) + body[pos + 1 :])
            )
        with pytest.raises(ValueError, match="truncated"):
            WaveformCodec.decompress(backend=backend, data=_reseal(body[:-1]))

    def test_lossless_samples_outside_int16(self) -> None:
        """An affine section decoding past 32767 is refused instead of wrapping."""
        section = struct.pack("!BiIB", 0, 32767, 1, 0) + wc._rice_encode(
            wc._zigzag(np.array([0, 1], dtype=np.int64))
        )
        with pytest.raises(ValueError, match="int16 range"):
            wc._decode_lossless_channel(section, 2)

    def test_rice_section_consistency_checks(self) -> None:
        """Wrong value counts, lengths and Rice parameters are refused."""
        values = np.array([0, 1, 5, 100, 3, 2**20], dtype=np.int64)
        encoded = wc._rice_encode(values)
        np.testing.assert_array_equal(wc._rice_decode(encoded, len(values)), values)
        with pytest.raises(ValueError, match="number of values"):
            wc._rice_decode(encoded, len(values) - 1)
        with pytest.raises(ValueError, match="wrong length"):
            wc._rice_decode(encoded + b"\x00", len(values))
        n_unary, n_rem, n_esc = struct.unpack("!III", encoded[:12])
        bad_rem = struct.pack("!III", n_unary, n_rem + 1, n_esc) + encoded[12:]
        with pytest.raises(ValueError, match="inconsistent|wrong length"):
            wc._rice_decode(bad_rem, len(values))
        bad_k = encoded[:12] + bytes([wc.WAVEFORM_CODEC_LOSSLESS_MAX_RICE_K + 1]) + encoded[13:]
        with pytest.raises(ValueError, match="Rice parameter out of range"):
            wc._rice_decode(bad_k, len(values))

    def test_table_index_outside_table(self) -> None:
        """A level index past the value table is refused."""
        section = struct.pack("!BI", 1, 1) + SpikeCodec._encode_varint(0) + bytes([0])
        section += wc._rice_encode(wc._zigzag(np.array([0, 3], dtype=np.int64)))
        with pytest.raises(ValueError, match="outside the value table"):
            wc._decode_lossless_channel(section, 2)

    @pytest.mark.parametrize("backend", ["python", "rust"])
    def test_zero_snippet_with_full_library_uses_template_zero(self, backend: str) -> None:
        """With the library full, a zero-norm snippet falls back to template 0, never to ID -1."""
        codec = WaveformCodec(snippet_samples=4, max_templates=1, backend=backend)
        first = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        templates, ids, scales, residuals = codec._template_match(
            [first, np.zeros(4, dtype=np.float32)]
        )
        assert len(templates) == 1 and ids == [0, 0] and scales == [1.0, 1.0]
        np.testing.assert_allclose(residuals[1], -first)

    @pytest.mark.parametrize("backend", ["python", "rust"])
    def test_scales_follow_the_best_of_several_templates(self, backend: str) -> None:
        """A snippet matching the first of two templates keeps that template and its scale."""
        codec = WaveformCodec(snippet_samples=4, max_templates=4, backend=backend)
        first = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        second = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
        templates, ids, scales, residuals = codec._template_match([first, second, 2.0 * first])
        assert len(templates) == 2 and ids == [0, 1, 0]
        assert scales == pytest.approx([1.0, 1.0, 2.0])
        np.testing.assert_allclose(residuals[2], 0.0, atol=1e-6)
        flat = np.zeros(4, dtype=np.float32)
        _, flat_ids, flat_scales, _ = codec._template_match([first, flat])
        assert flat_ids == [0, 1] and flat_scales == [1.0, 1.0], "a zero snippet matches nothing"

    def test_zigzag_inverts(self) -> None:
        """Zigzag mapping is a bijection on a symmetric range."""
        values = np.arange(-50, 51, dtype=np.int64)
        np.testing.assert_array_equal(wc._unzigzag(wc._zigzag(values)), values)
