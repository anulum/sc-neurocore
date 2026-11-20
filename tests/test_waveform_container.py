# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed waveform container interoperability tests

"""Exercise normal installed Python and Rust encoders and decoders end to end."""

from __future__ import annotations

import math
import struct
import sys
import zlib
from importlib.util import find_spec
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.spike_codec import WaveformCodec, describe_codec, probe_codec


def _recording(rows: int, channels: int) -> np.ndarray[Any, Any]:
    """Create an int16 trace with known clipped spike windows and a varying DC level."""
    values = (np.arange(rows)[:, None] % 29 - 17 + 40 * np.arange(channels)[None, :]).astype(
        np.int16
    )
    for row in (1, rows // 2, rows - 1):
        values[row] = np.arange(channels) * 100 - 12000
    return values


@pytest.mark.parametrize("mode", ("spike", "waveform", "full", "lossless", "lossless_rice"))
@pytest.mark.parametrize("geometry", ((13, 1), (259, 3), (1025, 7)))
def test_complete_container_roundtrip(
    mode: str, geometry: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both implementations write the same metadata and reconstruct the other's packet."""
    monkeypatch.setitem(sys.modules, "zstandard", None)
    monkeypatch.setitem(sys.modules, "pywt", None)
    rows, channels = geometry
    values = _recording(rows, channels)
    python, python_stats = WaveformCodec(mode=mode, backend="python", snippet_samples=13).compress(
        values
    )
    rust, rust_stats = WaveformCodec(mode=mode, backend="rust", snippet_samples=13).compress(
        values[:, ::-1].copy()[:, ::-1]
    )
    assert asdict(python_stats) == asdict(rust_stats)
    assert python == rust
    python_read = WaveformCodec.decompress(rust, max_samples=rows * channels, backend="python")
    rust_read = WaveformCodec.decompress(python, max_samples=rows * channels, backend="rust")
    assert python_read.mode == rust_read.mode == mode
    if mode.startswith("lossless"):
        assert python_read.lossless_waveform and rust_read.lossless_waveform
        assert python_read.spike_raster is rust_read.spike_raster is None
        assert python_read.waveform is not None and rust_read.waveform is not None
        assert python_read.waveform.dtype == rust_read.waveform.dtype == np.int16
        np.testing.assert_array_equal(python_read.waveform, values)
        np.testing.assert_array_equal(rust_read.waveform, values)
    else:
        assert python_read.spike_raster is not None and rust_read.spike_raster is not None
        np.testing.assert_array_equal(python_read.spike_raster, rust_read.spike_raster)
        if rows > 13:
            assert np.all(rust_read.spike_raster[[1, rows // 2, rows - 1]] == 1)
        if mode == "spike":
            assert python_read.waveform is rust_read.waveform is None
        else:
            assert rust_read.waveform is not None and python_read.waveform is not None
            assert rust_read.waveform.dtype == python_read.waveform.dtype == np.float32
            np.testing.assert_array_equal(python_read.waveform, rust_read.waveform)


def test_complete_native_scope_is_discoverable() -> None:
    """Discovery advertises the complete encoder and decoder for every mode."""
    descriptor = describe_codec("waveform")
    for variant in descriptor.variants:
        availability = probe_codec("waveform", variant.name)
        assert variant.native_scope == availability.native_scope == "container"
        assert availability.rust_components_available and "rust" in availability.backend_choices
        assert variant.native_symbols == ("waveform_container_encode", "waveform_container_decode")


def _sealed(body: bytes) -> bytes:
    """Finish a WFC2 header and sections with their checksum."""
    return body + struct.pack("!I", zlib.crc32(body))


def _replace_section(packet: bytes, index: int, edit: Callable[[bytes], bytes]) -> bytes:
    """Replace one length-prefixed section in a real encoded WFC2 packet."""
    start = 20
    for _ in range(index):
        start += 4 + struct.unpack_from("!I", packet, start)[0]
    end = start + 4 + struct.unpack_from("!I", packet, start)[0]
    section = edit(packet[start + 4 : end])
    return _sealed(packet[:start] + struct.pack("!I", len(section)) + section + packet[end:-4])


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_complete_decode_validates_framing_and_sample_limits(backend: str) -> None:
    """Both parsers check version, geometry, section bounds and the declared output limit."""
    packet, _ = WaveformCodec(mode="spike", backend="python").compress(_recording(259, 3))
    body = packet[:-4]
    for invalid, error in (
        (packet[:10], "shorter than"),
        (packet[:22], "header and checksum"),
        (b"ABCD" + packet[4:], "not a WaveformCodec"),
        (_sealed(body[:19] + bytes([5]) + body[20:]), "unknown WaveformCodec mode"),
        (_sealed(body[:-1]), "truncated"),
        (_sealed(body + b"x"), "trailing bytes"),
        (_replace_section(packet, 2, lambda _: b"x"), "spike mode requires empty"),
    ):
        with pytest.raises(ValueError, match=error):
            WaveformCodec.decompress(invalid, backend=backend)
    for limit in (-1, cast(int, 776.5), 776):
        with pytest.raises(ValueError, match="max_samples"):
            WaveformCodec.decompress(packet, max_samples=limit, backend=backend)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_complete_decode_validates_mode_specific_section_geometry(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject malformed background shape, scales and compressed sizes before using the samples."""
    monkeypatch.setitem(sys.modules, "zstandard", None)
    packet = WaveformCodec(mode="full", backend="python").compress(_recording(259, 3))[0]
    for edit, error in (
        (lambda _: b"", "background section"),
        (lambda bg: struct.pack("!I", 17) + bg[4:], "background dimensions"),
        (lambda bg: bg[:8] + struct.pack("!f", 0.0) + bg[12:], "delta scale"),
        (lambda bg: bg[:12] + struct.pack("!f", math.inf) + bg[16:], "background|non-finite"),
        (lambda bg: bg + b"x", "corrupted"),
        (lambda bg: bg[:24] + zlib.compress(b"\x00" * 46), "corrupted"),
        (lambda bg: bg[:24] + zlib.compress(b"\x01" + b"\x00" * 47), "background code"),
    ):
        with pytest.raises(ValueError, match=error):
            WaveformCodec.decompress(_replace_section(packet, 2, edit), backend=backend)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_legacy_spike_readback_and_lossless_formats_remain_distinct(backend: str) -> None:
    """The historical WFCX spike packet is readable and has no lossless variants."""
    values = _recording(259, 3)
    spike, _ = WaveformCodec(mode="spike", backend="python").compress(values)
    legacy = b"WFCX" + spike[4:-4]
    expected = WaveformCodec.decompress(spike, backend=backend).spike_raster
    np.testing.assert_array_equal(
        WaveformCodec.decompress(legacy, backend=backend).spike_raster, expected
    )
    for mode in ("full", "waveform", "lossless", "lossless_rice"):
        packet, _ = WaveformCodec(mode=mode, backend="python").compress(values)
        with pytest.raises(ValueError, match="legacy WFCX|only in the WFC2"):
            WaveformCodec.decompress(b"WFCX" + packet[4:-4], backend=backend)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_lossless_stream_header_refuses_contradicting_adc_codes(backend: str) -> None:
    """A declared one-bit ADC cannot silently quantise samples that share its codes."""
    values = np.asarray([[0], [1]], dtype=np.int16)
    with pytest.raises(ValueError, match="contradicts"):
        WaveformCodec(mode="lossless", adc_bits=1, backend=backend).compress(values)
    packet, _ = WaveformCodec(mode="lossless", adc_bits=16, backend=backend).compress(values)
    with pytest.raises(ValueError, match="invalid ADC width"):
        WaveformCodec.decompress(
            _replace_section(packet, 0, lambda body: b"\0" + body[1:]), backend=backend
        )


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_snippet_metadata_must_match_the_wire_geometry(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify template metadata, residual length and the odd-nibble padding contract."""
    monkeypatch.setitem(sys.modules, "zstandard", None)
    window = 13
    values = _recording(259, 3)
    packet = WaveformCodec(mode="waveform", backend="python", snippet_samples=window).compress(
        values
    )[0]
    templates, snippets = struct.unpack_from("!HH", packet, 12)
    assert 0 < templates < 128 and snippets == 9
    ids_at = 6 + templates * window + 4
    residuals_at = ids_at + 5 * snippets  # One-byte IDs followed by float32 scales.

    def with_wrong_padding(section: bytes) -> bytes:
        """Replace the unused low nibble in a valid compressed residual stream."""
        start = residuals_at + 8
        flat = zlib.decompress(section[start:])
        return section[:start] + zlib.compress(flat[:-1] + bytes([flat[-1] ^ 1]))

    def with_wrong_template_id(section: bytes) -> bytes:
        """Choose the index immediately beyond the carried template library."""
        return section[:ids_at] + bytes([templates]) + section[ids_at + 1 :]

    for edit, reason in (
        (lambda section: section[:2] + struct.pack("!f", -1.0) + section[6:], "template scale"),
        (with_wrong_template_id, "template ID"),
        (
            lambda section: section[: residuals_at + 4] + b"\0\0\0\0" + section[residuals_at + 8 :],
            "residual sample count",
        ),
        (with_wrong_padding, "residual padding"),
    ):
        with pytest.raises(ValueError, match=reason):
            WaveformCodec.decompress(_replace_section(packet, 1, edit), backend=backend)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_rice_affine_mapping_requires_a_positive_step(backend: str) -> None:
    """Disallow a non-invertible step while preserving a valid affine ramp."""
    values = np.arange(12, dtype=np.int16).reshape(-1, 1)
    packet, _ = WaveformCodec(mode="lossless_rice", backend="python").compress(values)
    np.testing.assert_array_equal(
        WaveformCodec.decompress(packet, backend=backend).waveform, values
    )

    def without_step(body: bytes) -> bytes:
        """Set the uint32 affine step to zero in the only channel."""
        assert body[4] == 0
        return body[:9] + b"\0\0\0\0" + body[13:]

    with pytest.raises(ValueError, match="step must be positive"):
        WaveformCodec.decompress(_replace_section(packet, 0, without_step), backend=backend)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_decoders_refuse_conflicting_event_counts(backend: str) -> None:
    """Reject inconsistent template, event and section declarations before reconstruction."""
    samples = _recording(259, 3)
    spike = WaveformCodec(mode="spike", backend="python").compress(samples)[0][:-4]
    waveform = WaveformCodec(mode="waveform", backend="python").compress(samples)[0]
    lossless = WaveformCodec(mode="lossless", backend="python").compress(samples)[0][:-4]

    def counts(body: bytes, templates: int, events: int) -> bytes:
        """Change the fixed header counts and recompute its checksum."""
        return _sealed(body[:12] + struct.pack("!HH", templates, events) + body[16:])

    for invalid, reason in (
        (counts(spike, 64, 9), "template count"),
        (counts(spike, 0, 259 * 3 + 1), "snippet count"),
        (counts(spike, 0, 8), "spike raster"),
        (counts(lossless, 0, 1), "zero spike counts"),
        (_replace_section(waveform, 2, lambda _: b"x"), "empty background"),
        (_replace_section(waveform, 1, lambda _: b""), "malformed|truncated"),
    ):
        with pytest.raises(ValueError, match=reason):
            WaveformCodec.decompress(invalid, backend=backend)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_snippet_section_requires_complete_counts_and_varints(backend: str) -> None:
    """Check contradictory counts, unrepresentable IDs and unexpected zero-event data."""
    values = _recording(259, 3)
    packet = WaveformCodec(mode="waveform", backend="python", snippet_samples=13).compress(values)[
        0
    ]
    templates, snippets = struct.unpack_from("!HH", packet, 12)
    counts_end = 6 + 13 * templates + 4
    scales_start = counts_end + snippets
    quiet = WaveformCodec(mode="waveform", backend="python").compress(np.zeros((16, 1)))[0]
    for encoded, edit, reason in (
        (packet, lambda s: struct.pack("!H", 0) + s[2:], "template count"),
        (packet, lambda s: s[: counts_end - 4] + b"\0" * 4 + s[counts_end:], "snippet count"),
        (packet, lambda s: s[:counts_end] + b"\x80", "varint|truncated"),
        (packet, lambda s: s[:counts_end] + b"\xff\xff\x04", "varint|range"),
        (packet, lambda s: s[:counts_end] + b"\x80\x80\x80\0", "uint16"),
        (
            packet,
            lambda s: s[:scales_start] + struct.pack("!f", math.inf) + s[scales_start + 4 :],
            "non-finite|scale",
        ),
        (quiet, lambda s: s + b"x", "trailing bytes"),
    ):
        with np.errstate(invalid="ignore"), pytest.raises(ValueError, match=reason):
            WaveformCodec.decompress(_replace_section(encoded, 1, edit), backend=backend)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_full_background_refuses_overflow_in_channel_reconstruction(backend: str) -> None:
    """Finite first-row differences may still produce an unrepresentable signal."""
    samples = np.zeros((3, 2), dtype=np.int16)
    packet = WaveformCodec(mode="full", backend="python").compress(samples)[0]

    def with_large_channel_differences(section: bytes) -> bytes:
        """Replace both first-row differences with the largest finite float32."""
        limit = float(np.finfo(np.float32).max)
        return section[:12] + struct.pack("!ff", limit, limit) + section[20:]

    with np.errstate(over="ignore"), pytest.raises(ValueError, match="non-finite.*background"):
        WaveformCodec.decompress(
            _replace_section(packet, 2, with_large_channel_differences), backend=backend
        )


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_rice_mapping_and_signed_predictor_bounds(backend: str) -> None:
    """Invalid level maps and predictors must fail before casting back to int16."""
    values = np.resize(np.array([1, 1000, 3000, 32761], dtype=np.int16), (512, 1))
    indexed = WaveformCodec(mode="lossless_rice", backend="python").compress(values)[0]
    ramp = WaveformCodec(mode="lossless_rice", backend="python").compress(
        np.arange(12, dtype=np.int16).reshape(-1, 1)
    )[0]
    singleton = WaveformCodec(mode="lossless_rice", backend="python").compress(
        np.zeros((1, 1), dtype=np.int16)
    )[0]

    def edit_table(body: bytes, entries: bytes) -> bytes:
        """Replace the leading one-byte table entry or introduce a duplicate."""
        assert body[4] == 1 and body[9] == 2
        channel = body[4:9] + entries + body[10:]
        return struct.pack("!I", len(channel)) + channel

    def edit_predictor(body: bytes) -> bytes:
        """Set the first level of an order-two predictor outside the allowed range."""
        assert body[4] == 0 and body[13:15] == b"\x02\0"
        channel = body[4:14] + b"\x80\x80\x08" + body[15:]
        return struct.pack("!I", len(channel)) + channel

    def edit_quotient(body: bytes) -> bytes:
        """Use a unary quotient of 33, one above the reserved escape code."""
        (channel_size,) = struct.unpack_from("!I", body)
        assert channel_size == len(body) - 4
        mapping_size = 9 if body[4] == 0 else 6
        prefix = body[4 : 5 + mapping_size]
        assert prefix[-1] == 0
        channel = prefix + struct.pack("!III", 34, 0, 0) + b"\0\xff\xff\xff\xff\x80"
        return struct.pack("!I", len(channel)) + channel

    for packet, edit, reason in (
        (indexed, lambda s: edit_table(s, b"\x80\x80\x04"), "table sample.*int16"),
        (indexed, lambda s: edit_table(s, b"\x02\0"), "strictly increasing"),
        (ramp, edit_predictor, "predictor.*range|outside.*range"),
        (singleton, edit_quotient, "quotient|wrong number of values"),
    ):
        with pytest.raises(ValueError, match=reason):
            WaveformCodec.decompress(_replace_section(packet, 0, edit), backend=backend)


def test_zstd_frame_must_match_background_geometry() -> None:
    """A valid zstd frame with a mismatched declared size must be refused."""
    values = np.zeros((64, 2), dtype=np.int16)
    packet = WaveformCodec(mode="full", backend="python").compress(values)[0]
    # Zstandard frame with seven zero bytes; the 4 × 2 background needs eight.
    short = bytes.fromhex("28b52ffd200739000000000000000000")
    invalid = _replace_section(packet, 2, lambda s: s[:20] + short)
    for backend in ("python", "rust"):
        reason = "frame size|section.*length"
        if backend == "python" and find_spec("zstandard") is None:
            reason = "needs the zstandard package"
        with pytest.raises(ValueError, match=reason):
            WaveformCodec.decompress(invalid, backend=backend)


def test_automatic_dispatch_with_a_kernel_only_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the installed kernels when the engine exposes only the preceding API."""
    from sc_neurocore_engine import sc_neurocore_engine as engine

    samples = _recording(259, 3)
    monkeypatch.delattr(engine, "waveform_container_encode")
    monkeypatch.delattr(engine, "waveform_container_decode")
    for mode in ("spike", "waveform", "full", "lossless", "lossless_rice"):
        expected, _ = WaveformCodec(mode=mode, backend="python").compress(samples)
        actual, _ = WaveformCodec(mode=mode, backend="auto").compress(samples)
        assert expected == actual
        reference = WaveformCodec.decompress(expected, backend="python")
        decoded = WaveformCodec.decompress(actual, backend="auto")
        np.testing.assert_array_equal(decoded.spike_raster, reference.spike_raster)
        np.testing.assert_array_equal(decoded.waveform, reference.waveform)
        with pytest.raises(ValueError, match="complete WaveformCodec"):
            WaveformCodec.decompress(actual, backend="rust")


def test_installed_dependency_errors_are_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Distinguish an absent package from a present package with a broken import."""
    missing = "invalid_waveform_transitive_dependency"
    packet_input = _recording(259, 3)
    broken_optional = tmp_path / "zstandard.py"
    broken_optional.write_text(f"import {missing}\n", encoding="utf-8")
    with monkeypatch.context() as environment:
        environment.syspath_prepend(str(tmp_path))
        environment.delitem(sys.modules, "zstandard", raising=False)
        with pytest.raises(ModuleNotFoundError, match=missing):
            WaveformCodec(mode="full", backend="rust").compress(packet_input)

    engine = tmp_path / "sc_neurocore_engine"
    engine.mkdir()
    (engine / "__init__.py").write_text("", encoding="utf-8")
    (engine / "sc_neurocore_engine.py").write_text(f"import {missing}\n", encoding="utf-8")
    with monkeypatch.context() as environment:
        environment.syspath_prepend(str(tmp_path))
        environment.delitem(sys.modules, "sc_neurocore_engine")
        environment.delitem(sys.modules, "sc_neurocore_engine.sc_neurocore_engine")
        with pytest.raises(ModuleNotFoundError, match=missing):
            WaveformCodec(backend="auto")
