# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ISI raster codec interoperability

"""Verify wire compatibility and backend selection through the public codec."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from itertools import product
from typing import Any

import numpy as np
import pytest

from sc_neurocore.spike_codec import SpikeCodec, probe_codec

_BACKENDS = probe_codec("isi").backend_choices


def test_auto_uses_the_installed_packet_codec() -> None:
    """Hosted native jobs require a real Rust implementation for both directions."""
    expected = "rust" if "rust" in _BACKENDS else "python"
    assert SpikeCodec().backend == expected
    if os.environ.get("SC_NEUROCORE_REQUIRE_ENGINE") == "1":
        assert SpikeCodec(backend="rust").backend == "rust"
    elif expected == "python":
        with pytest.raises(ValueError, match="needs sc_neurocore_engine"):
            SpikeCodec(backend="rust")


def test_real_import_refusal_agrees_with_the_catalogue() -> None:
    """A present native engine refused by the host must not silently fall back."""
    if "rust" not in _BACKENDS:
        assert probe_codec("isi").backend_choices == ("auto", "python")
        return

    program = """
import sys

attempts = []
def refuse(event, arguments):
    if event == "import" and arguments[0] == "sc_neurocore_engine.sc_neurocore_engine":
        attempts.append(arguments[0])
        raise ImportError("denied by the host")

sys.addaudithook(refuse)
from sc_neurocore.spike_codec import SpikeCodec, probe_codec

assert probe_codec("isi").backend_choices == ("python",)
try:
    SpikeCodec()
except ImportError:
    pass
else:
    raise AssertionError("the present native engine fell back after import denial")
assert attempts
assert SpikeCodec(backend="python").backend == "python"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "mode,precision,entropy,backend,field",
    [
        ("unknown", 1, "auto", "python", "mode"),
        ("lossless", 1, "unknown", "python", "entropy"),
        ("lossy", 0, "auto", "python", "timing_precision"),
        ("lossless", 1, "auto", "unknown", "backend"),
    ],
)
def test_invalid_codec_configuration(
    mode: str, precision: int, entropy: str, backend: str, field: str
) -> None:
    """Bad configuration raises at public construction."""
    with pytest.raises(ValueError, match=field):
        SpikeCodec(mode, precision, entropy, backend=backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_revalidate_mutated_packet_configuration(backend: str) -> None:
    """Mutable configuration cannot encode an invalid timing or entropy setting."""
    codec = SpikeCodec("lossy", 2, backend=backend)
    raster = np.ones((8, 1), dtype=np.int8)
    codec.timing_precision = 0
    with pytest.raises(ValueError, match="timing_precision"):
        codec.compress(raster)
    codec.timing_precision = 2
    codec.entropy = "unknown"
    with pytest.raises(ValueError, match="entropy"):
        codec.compress(raster)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_header_field_width(backend: str) -> None:
    """The caller's row count must fit in the packet's uint32 field."""
    codec = SpikeCodec(backend=backend)
    with pytest.raises(ValueError, match="uint32"):
        codec.decompress(b"", 1 << 32, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("rows", [1 << 25, (1 << 32) - 1])
def test_legacy_varint_packet_with_large_row_count(backend: str, rows: int) -> None:
    """Read legacy varint packets whose leading byte is distinct from Huffman."""
    codec = SpikeCodec("lossy", rows // 2, backend=backend)
    packet = struct.pack("!II", rows, 2) + bytes((2, 0, 1, 1, 1))
    raster = codec.decompress(packet, 2, 2)
    np.testing.assert_array_equal(raster, np.array([[1, 0], [1, 1]], dtype=np.int8))


def _raster(pattern: str) -> np.ndarray[Any, Any]:
    if pattern == "empty":
        return np.empty((0, 3), dtype=np.int8)
    if pattern == "no_channels":
        return np.empty((8, 0), dtype=np.int8)
    if pattern == "sparse":
        raster = np.zeros((181, 3), dtype=np.int8)
        raster[[0, 129, 160], 1] = 1
        raster[180, 2] = 1
        return raster
    raster = np.zeros((181, 6), dtype=bool)
    if pattern == "bursts":
        raster[2:39, 0] = True
        raster[73:119, 2] = True
        return raster[:, ::2]
    return np.ones((31, 3), dtype=bool, order="F")


@pytest.mark.parametrize("pattern", ["empty", "no_channels", "sparse", "bursts", "dense"])
@pytest.mark.parametrize("entropy", ["varint", "huffman", "auto"])
@pytest.mark.parametrize("precision", [1, 7, 256])
def test_backend_packets_and_cross_decoding(pattern: str, entropy: str, precision: int) -> None:
    """Every available backend emits the same packet and accepts the other packets."""
    raster = _raster(pattern)
    rows, channels = raster.shape
    bins = rows // precision
    expected = np.zeros((rows, channels), dtype=np.int8)
    for bin_index in range(bins):
        start = bin_index * precision
        expected[bin_index] = raster[start : start + precision].any(axis=0)
    codecs = {
        backend: SpikeCodec(
            mode="lossless" if precision == 1 else "lossy",
            timing_precision=precision,
            entropy=entropy,
            backend=backend,
        )
        for backend in _BACKENDS
    }
    packets = {backend: codec.compress(raster) for backend, codec in codecs.items()}
    reference = packets["python"]
    assert reference[1].n_spikes == np.count_nonzero(expected)
    for encoder, decoder in product(codecs, repeat=2):
        assert packets[encoder] == reference
        packet = packets[encoder][0]
        codec = codecs[decoder]
        np.testing.assert_array_equal(codec.decompress(packet, rows, channels), expected)
        np.testing.assert_array_equal(codec.decompress(packet, bins, channels), expected[:bins])


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("entropy", ["varint", "huffman"])
def test_packet_framing_refuses_truncation_and_shape_mismatch(backend: str, entropy: str) -> None:
    """Every incomplete prefix, extra byte and conflicting shape fails at the public decoder."""
    raster = _raster("sparse")
    codec = SpikeCodec(entropy=entropy, backend=backend)
    packet, _ = codec.compress(raster)
    for prefix in range(len(packet)):
        with pytest.raises(ValueError):
            codec.decompress(packet[:prefix], *raster.shape)
    with pytest.raises(ValueError, match="trailing"):
        codec.decompress(packet + b"\x00", *raster.shape)
    with pytest.raises(ValueError, match="dimensions"):
        codec.decompress(packet, raster.shape[0] + 1, raster.shape[1])
    with pytest.raises(ValueError, match="dimensions"):
        codec.decompress(packet, raster.shape[0], raster.shape[1] + 1)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "channel",
    [
        b"\x01\x08",  # first time outside the raster
        b"\x02\x00\x00",  # duplicate timestamp
        b"\x02\x01\x07",  # out-of-range cumulative time
        b"\x09",  # event count greater than the row count
        b"\x81\x00\x02",  # nonminimal event count
        b"\x80\x80\x80\x80\x10",  # uint32 overflow
    ],
)
def test_varint_fields_refuse_invalid_values(backend: str, channel: bytes) -> None:
    """Invalid counts, timestamps and LEB128 representations raise ValueError."""
    packet = struct.pack("!II", 8, 1) + channel
    with pytest.raises(ValueError):
        SpikeCodec(backend=backend).decompress(packet, 8, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "table,bits,body",
    [
        ([(1, 1), (1, 2)], 3, b"\x00"),
        ([(1, 0)], 3, b"\x00"),
        ([(1, 64)], 3, b"\x00"),
        ([(1, 1), (2, 1), (3, 1)], 3, b"\x00"),
        ([(1, 1)], 3, b"\x01"),
        ([(1, 1)], 3, b"\x80"),
        ([(1, 1)], 4, b"\x00"),
        ([(1, 1)], 64, b"\xff" * 8),
        ([(0, 1)], 3, b"\x00"),
        ([(-1, 1)], 3, b"\x00"),
    ],
)
def test_huffman_fields_refuse_invalid_values(
    backend: str, table: list[tuple[int, int]], bits: int, body: bytes
) -> None:
    """Invalid canonical tables, padding, codes and intervals fail during decode."""
    packet = (
        b"\x01"
        + struct.pack("!III", 8, 1, 1)
        + b"\x04"
        + struct.pack("!I", 1)
        + b"\x00"
        + struct.pack("!H", len(table))
        + b"".join(struct.pack("!iB", symbol, bits) for symbol, bits in table)
        + struct.pack("!I", bits)
        + body
    )
    with pytest.raises(ValueError):
        SpikeCodec(backend=backend).decompress(packet, 8, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_huffman_count_section_requires_one_count_per_channel(backend: str) -> None:
    """A complete section of the wrong cardinality is not a complete packet."""
    packet = b"\x01" + struct.pack("!IIII", 8, 1, 0, 0) + b"\x00\x00"
    with pytest.raises(ValueError, match="count section"):
        SpikeCodec(backend=backend).decompress(packet, 8, 1)
