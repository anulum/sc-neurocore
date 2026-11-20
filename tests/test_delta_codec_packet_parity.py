# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Delta codec packet parity and geometry

"""Exercise complete DSCX packets through both public codec implementations."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.spike_codec import (
    DeltaSpikeCodec,
    SpikeCodec,
    delta_codec,
    get_codec,
    probe_codec,
)
from tests.engine_requirement import require_engine


@pytest.mark.parametrize(
    ("rows", "channels", "size"), [(13, 9, 4), (7, 513, 256), (9, 3, 1), (0, 3, 2), (4, 0, 4)]
)
def test_native_and_python_packets_match(rows: int, channels: int, size: int) -> None:
    """Retain packet identity, counters and bidirectional decoding across group boundaries."""
    require_engine()
    values = (np.random.default_rng(42).random((channels, rows)) < 0.35).T
    reference = get_codec("delta", group_size=size, backend="python")
    native = DeltaSpikeCodec(group_size=size, backend="rust")
    assert native.backend == "rust"
    assert DeltaSpikeCodec().backend == "rust"
    assert probe_codec("delta").backend_choices == ("auto", "python", "rust")
    python_packet, python_result = reference.compress(values)
    native_packet, native_result = native.compress(values)
    assert native_packet == python_packet
    assert native_result == python_result
    np.testing.assert_array_equal(native.decompress(python_packet, rows, channels), values)
    np.testing.assert_array_equal(reference.decompress(native_packet, rows, channels), values)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_group_references_use_first_tie_and_last_valid_index(backend: str) -> None:
    """Store the first maximum on ties and index 255 in a full group."""
    if backend == "rust":
        require_engine()
    raster = np.zeros((5, 513), dtype=np.int8)
    raster[:, 2] = raster[:, 3] = raster[:, 511] = 1
    packet, result = DeltaSpikeCodec(group_size=256, backend=backend).compress(raster)
    assert packet[:11] == b"DSCX" + struct.pack("!HH", 256, 3) + bytes([2, 255, 0])
    assert result.n_spikes == 15
    assert result.mean_delta_sparsity == pytest.approx(1.0 - 509 / 510)
    np.testing.assert_array_equal(
        DeltaSpikeCodec(backend=backend).decompress(packet, 5, 513), raster
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_lossy_residuals_preserve_complete_bin_semantics(backend: str) -> None:
    """Quantise references and XOR residuals before restoring each compact bin."""
    if backend == "rust":
        require_engine()
    raster = np.array([[1, 1], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.int8)
    codec = DeltaSpikeCodec(group_size=2, mode="lossy", timing_precision=2, backend=backend)
    packet, result = codec.compress(raster)
    assert not result.lossless
    assert packet[:9] == b"DSCX" + struct.pack("!HH", 2, 1) + b"\x01"
    expected_residual = np.array([[1, 1], [1, 1]], dtype=np.int8)
    residual = SpikeCodec(mode="lossy", timing_precision=2, backend="python")
    np.testing.assert_array_equal(residual.decompress(packet[9:], 2, 2), expected_residual)
    expected = np.array([[0, 1], [0, 1]], dtype=np.int8)
    np.testing.assert_array_equal(codec.decompress(packet, 2, 2), expected)
    np.testing.assert_array_equal(codec.decompress(packet, 5, 2)[:2], expected)
    reference = DeltaSpikeCodec(group_size=2, mode="lossy", timing_precision=2, backend="python")
    assert packet == reference.compress(raster)[0]


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_reader_rejects_invalid_group_geometry_and_body(backend: str) -> None:
    """Refuse mismatched counts, out-of-group references and malformed ISI bodies."""
    if backend == "rust":
        require_engine()
    codec = DeltaSpikeCodec(group_size=4, backend=backend)
    packet, _ = codec.compress(np.zeros((5, 9), dtype=np.int8))
    cases = (
        (packet[:5], "truncated DSCX header"),
        (packet[:4] + b"\x00\x00" + packet[6:], "group_size must be in"),
        (packet[:6] + b"\x00\x02" + packet[8:], "group count does not match"),
        (packet[:10], "truncated DSCX references"),
        (packet[:8] + b"\x04" + packet[9:], "outside its group"),
        (packet[:10] + b"\x01" + packet[11:], "outside its group"),
        (packet[:-1], "ISI"),
    )
    for invalid, message in cases:
        with pytest.raises(ValueError, match=message):
            codec.decompress(invalid, 5, 9)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_writer_rejects_nonbinary_and_unrepresentable_inputs(backend: str) -> None:
    """Refuse nonbinary values before narrowing and counts beyond the wire width."""
    if backend == "rust":
        require_engine()
    codec = DeltaSpikeCodec(group_size=1, backend=backend)
    for dtype in (np.int8, np.int64, np.float64):
        with pytest.raises(ValueError, match="zeros and ones"):
            codec.compress(np.array([[2]], dtype=dtype))
    with pytest.raises(ValueError, match="group count exceeds uint16"):
        codec.compress(np.empty((0, 65536), dtype=np.int8))


def test_invalid_backend_is_refused() -> None:
    """Reject unknown backend names before encoding."""
    with pytest.raises(ValueError, match="backend must be auto, python or rust"):
        DeltaSpikeCodec(backend="unavailable")


def test_delta_backend_selection_when_native_bindings_are_absent() -> None:
    """Exercise real fallback and explicit Rust refusal in Python-only or older installs."""
    availability = probe_codec("delta")
    if availability.rust_components_available:
        assert get_codec("delta").backend == "rust"
    else:
        assert get_codec("delta").backend == "python"
        with pytest.raises(ValueError, match="needs sc_neurocore_engine with the DSCX codec"):
            get_codec("delta", backend="rust")


@pytest.mark.parametrize("error", ["ImportError", "RuntimeError"])
def test_delta_distinguishes_denied_native_loading(error: str) -> None:
    """Report a real host admission failure and keep explicit Python usable."""
    require_engine()
    assert delta_codec.__file__ is not None
    module_path = Path(delta_codec.__file__).resolve()
    program = """
import builtins
from pathlib import Path
import sys

expected = Path(sys.argv[1])
sys.path.insert(0, str(expected.parents[2]))
exception = getattr(builtins, sys.argv[2])
message = "embedding policy denied the native codec extension"
attempts = []

def deny_extension(event, arguments):
    if event == "import" and arguments[0] == "sc_neurocore_engine.sc_neurocore_engine":
        attempts.append(arguments[0])
        raise exception(message)

sys.addaudithook(deny_extension)
import numpy as np
from sc_neurocore.spike_codec import DeltaSpikeCodec, delta_codec, probe_codec

assert Path(delta_codec.__file__).resolve() == expected
status = probe_codec("delta")
assert status.backend_choices == ("python",)
assert attempts
assert any(exception.__name__ in diagnostic for diagnostic in status.diagnostics)
try:
    DeltaSpikeCodec()
except exception as failure:
    assert str(failure)
else:
    raise AssertionError("auto silently accepted a broken native installation")

raster = np.array([[1, 1], [0, 1]], dtype=np.int8)
codec = DeltaSpikeCodec(backend="python")
packet, _ = codec.compress(raster)
np.testing.assert_array_equal(codec.decompress(packet, *raster.shape), raster)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(module_path), error],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
