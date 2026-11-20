# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Address-event wire parity and refusals

"""Verify legacy address-event packets through both installed backends."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.spike_codec import AERSpikeCodec, aer_codec, get_codec, probe_codec
from tests.engine_requirement import require_engine

_PACKET = (
    b"AERX\x00\x00\x00\x14\x00\x00\x00\x03\x00\x00\x00\x03\x01"
    b"\x00\x00\x00\x00\x00\x02\x00\x07\xff\x00\x07\xff\x00\x04\x01"
)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_known_simultaneous_events_and_escapes(backend: str) -> None:
    """Decode ordered simultaneous spikes and preserve repeated escape records."""
    if backend == "rust":
        require_engine()
    raster = np.zeros((20, 3), dtype=np.int8)
    raster[0, [0, 2]] = 1
    raster[18, 1] = 1
    codec = AERSpikeCodec(timestamp_bits=3, backend=backend)
    packet, result = codec.compress(raster)
    assert packet == _PACKET
    assert (result.n_events, result.n_spikes, result.bytes_per_event) == (3, 3, len(packet) / 3)
    np.testing.assert_array_equal(codec.decompress(packet), raster)
    np.testing.assert_array_equal(codec.decompress(packet, T=1, N=2), raster[:1, :2])


@pytest.mark.parametrize(
    "shape,density,neuron_bits,expected",
    [
        ((0, 7), 0.0, 0, b"AERX"),
        ((9, 0), 0.0, 0, b"AERX"),
        ((9, 255), 0.2, 0, b"AERX"),
        ((9, 256), 0.8, 8, b"AERI"),
        ((9, 17), 1.0, 2040, b"AERI"),
        ((19, 15), 0.3, 40, b"AERX"),
    ],
)
def test_native_and_reference_exchange_packets(
    shape: tuple[int, int], density: float, neuron_bits: int, expected: bytes
) -> None:
    """Preserve both polarities, empty dimensions and reserved address widths."""
    require_engine()
    values = (np.random.default_rng(42).random(shape[::-1]) < density).T
    python = AERSpikeCodec(neuron_bits=neuron_bits, backend="python")
    rust = get_codec("aer", neuron_bits=neuron_bits, backend="rust")
    packet, stats = python.compress(values)
    assert packet[:4] == expected and rust.backend == "rust"
    assert rust.compress(values) == (packet, stats)
    for codec in (python, rust):
        result = codec.decompress(packet)
        assert result.dtype == np.int8
        np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_packet_refuses_inconsistent_header_and_event_records(backend: str) -> None:
    """Reject truncated, unordered, duplicate, overflowing and dangling records."""
    if backend == "rust":
        require_engine()
    codec = AERSpikeCodec(backend=backend)
    head = _PACKET[:17]
    cases = (
        (head[:-1], "truncated AER header"),
        (head[:-1] + b"\x00", "address width"),
        (head[:12] + struct.pack("!I", 61) + head[16:], "event count"),
        (_PACKET[:-1], "body length"),
        (_PACKET + b"\x00\x00\x00", "trailing"),
        (head + b"\x00\x00\xff" * 3, "must advance"),
        (head + b"\x00\x00\x00" * 3, "must increase"),
        (head + b"\x00\x00\x03" * 3, "address is outside"),
        (head + b"\x00\x14\x00" * 3, "timestamp is outside"),
        (head + b"\x00\x01\xff" * 3, "incomplete"),
        (head[:-1] + b"\x05" + b"\x00\x00\x01\x00\x00\x00\x00" * 3, "address exceeds"),
        (
            b"AERX"
            + struct.pack("!IIIB", 0xFFFFFFFF, 1, 1, 1)
            + b"\x00\x01\xff"
            + b"\xff\xff\xff" * 65537,
            "timestamp overflows",
        ),
    )
    for packet, message in cases:
        with pytest.raises(ValueError, match=message):
            codec.decompress(packet)
    with pytest.raises(ValueError, match="shape exceeds"):
        codec.decompress(_PACKET, N=4)
    with pytest.raises(ValueError, match="only zeros and ones"):
        codec.compress(np.array([[2]], dtype=np.int8))


def test_explicit_backend_and_representable_settings() -> None:
    """Keep automatic fallback on older packages and validate mutable field settings."""
    status = probe_codec("aer")
    if status.rust_components_available:
        assert AERSpikeCodec().backend == "rust"
        assert status.backend_choices == ("auto", "python", "rust")
    else:
        assert AERSpikeCodec().backend == "python"
        assert status.backend_choices == ("auto", "python")
        with pytest.raises(ValueError, match="needs sc_neurocore_engine"):
            AERSpikeCodec(backend="rust")
    codec = AERSpikeCodec(backend="python")
    for timestamp, neuron in ((0, 0), (17, 0), (16, 2041)):
        with pytest.raises(ValueError, match="bit width"):
            AERSpikeCodec(timestamp, neuron)
    with pytest.raises(ValueError, match="backend must be"):
        AERSpikeCodec(backend="unavailable")
    with pytest.raises(ValueError, match="shape"):
        codec.compress(np.array([1], dtype=np.int8))
    codec.timestamp_bits = 0
    with pytest.raises(ValueError, match="bit width"):
        codec.compress(np.ones((3, 2), dtype=np.int8))


@pytest.mark.parametrize("error", ["ImportError", "RuntimeError"])
def test_reports_denied_native_event_loading(error: str) -> None:
    """Apply a real process loader policy while leaving explicit Python usable."""
    require_engine()
    assert aer_codec.__file__ is not None
    expected = Path(aer_codec.__file__).resolve()
    program = """
import builtins
from pathlib import Path
import sys

expected = Path(sys.argv[1])
sys.path.insert(0, str(expected.parents[2]))
exception = getattr(builtins, sys.argv[2])
attempts = []

def deny(event, arguments):
    if event == "import" and arguments[0] == "sc_neurocore_engine.sc_neurocore_engine":
        attempts.append(arguments[0])
        raise exception("host denied address-event extension loading")

sys.addaudithook(deny)
import numpy as np
from sc_neurocore.spike_codec import AERSpikeCodec, aer_codec, probe_codec

assert Path(aer_codec.__file__).resolve() == expected
status = probe_codec("aer")
assert attempts and status.backend_choices == ("python",)
assert any(exception.__name__ in item for item in status.diagnostics)
try:
    AERSpikeCodec()
except exception:
    pass
else:
    raise AssertionError("automatic selection hid a broken native installation")
codec = AERSpikeCodec(backend="python")
raster = np.array([[1, 0], [0, 1]], dtype=np.int8)
np.testing.assert_array_equal(codec.decompress(codec.compress(raster)[0]), raster)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(expected), error],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
