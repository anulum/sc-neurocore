# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Raster packet parity and wire invariants

"""Exercise SSCF and standalone packets through the public codec."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.spike_codec import StreamingSpikeCodec, get_codec, probe_codec, streaming_codec
from tests.engine_requirement import require_engine

_RASTER = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
_FRAME = b"\x00\x03\x00\x03\x02\x05\x02"
_CONTAINER = (
    b"SSCF\x00\x02\x00\x00\x00\x03\x00\x03\x00\x00\x00\x02"
    b"\x00\x03\x00\x02\x02\x01\x02"
    b"\x00\x03\x00\x02\x06\x01"
)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_matches_known_frames_and_legacy_container(backend: str) -> None:
    """Verify bit order, frame padding, output prefix and complete statistics."""
    if backend == "rust":
        require_engine()
    codec = StreamingSpikeCodec(2, backend=backend)
    assert codec.compress_frame(_RASTER) == _FRAME
    np.testing.assert_array_equal(codec.decompress_frame(_FRAME), _RASTER)
    packet, stats = codec.compress(_RASTER)
    assert packet == _CONTAINER
    assert (stats.n_frames, stats.n_spikes, stats.max_frame_bytes) == (2, 3, 7)
    assert stats.mean_active_channels == 1.5
    np.testing.assert_array_equal(codec.decompress(_CONTAINER), _RASTER)
    np.testing.assert_array_equal(codec.decompress(_CONTAINER, T=2, N=3), _RASTER[:2])


@pytest.mark.parametrize(
    "shape,window",
    [((0, 0), 1), ((0, 11), 9), ((17, 0), 8), ((17, 19), 8), ((37, 65), 16), ((5, 2), 65535)],
)
def test_native_and_python_exchange_packets(shape: tuple[int, int], window: int) -> None:
    """Exchange exact packets for non-contiguous input, empty axes and window bounds."""
    require_engine()
    rows, channels = shape
    raster = (np.random.default_rng(112).random((channels, rows)) < 0.2).T
    reference = StreamingSpikeCodec(window, backend="python")
    native = get_codec("streaming", window_size=window, backend="rust")
    assert native.backend == "rust"
    packet, stats = reference.compress(raster)
    assert native.compress(raster) == (packet, stats)
    frame = native.compress_frame(raster)
    assert frame == reference.compress_frame(raster)
    for codec in (reference, native):
        decoded = codec.decompress(packet)
        np.testing.assert_array_equal(decoded, raster)
        assert decoded.dtype == np.int8
        np.testing.assert_array_equal(codec.decompress_frame(frame), raster)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_refuses_incomplete_or_inconsistent_packets(backend: str) -> None:
    """Refuse every truncated prefix, trailing bytes, changed geometry and pad spikes."""
    if backend == "rust":
        require_engine()
    codec = StreamingSpikeCodec(backend=backend)
    for cut in range(len(_FRAME)):
        with pytest.raises(ValueError):
            codec.decompress_frame(_FRAME[:cut])
    for cut in range(len(_CONTAINER)):
        with pytest.raises(ValueError):
            codec.decompress(_CONTAINER[:cut])
    with pytest.raises(ValueError, match="trailing raster frame bytes"):
        codec.decompress_frame(_FRAME + b"\x00")
    with pytest.raises(ValueError, match="trailing SSCF bytes"):
        codec.decompress(_CONTAINER + b"\x00")
    with pytest.raises(ValueError, match="nonzero raster frame padding"):
        codec.decompress_frame(_FRAME[:-1] + b"\x82")
    with pytest.raises(ValueError, match="nonzero raster frame padding"):
        codec.decompress(_CONTAINER[:-1] + b"\x03")
    malformed = _CONTAINER[:18] + struct.pack("!H", 1) + _CONTAINER[20:]
    with pytest.raises(ValueError, match="dimensions do not match SSCF header"):
        codec.decompress(malformed)
    with pytest.raises(ValueError, match="invalid SSCF frame count or window size"):
        codec.decompress(_CONTAINER[:4] + b"\x00\x00" + _CONTAINER[6:])
    with pytest.raises(ValueError, match="does not match SSCF header"):
        codec.decompress(_CONTAINER, T=4)
    with pytest.raises(ValueError, match="does not match SSCF header"):
        codec.decompress(_CONTAINER, N=2)


def test_catalogue_uses_only_a_complete_native_backend() -> None:
    """Select the Rust packet path only when all four native operations exist."""
    status = probe_codec("streaming")
    if status.rust_components_available:
        assert status.backend_choices == ("auto", "python", "rust")
        assert StreamingSpikeCodec().backend == "rust"
    else:
        assert status.backend_choices == ("auto", "python")
        assert StreamingSpikeCodec().backend == "python"
        with pytest.raises(ValueError, match="needs sc_neurocore_engine"):
            StreamingSpikeCodec(backend="rust")


def test_refuses_invalid_rasters_before_narrowing() -> None:
    """Apply the documented field widths and binary input contract."""
    codec = StreamingSpikeCodec(backend="python")
    with pytest.raises(ValueError, match="1..65535"):
        StreamingSpikeCodec(0)
    with pytest.raises(ValueError, match="backend must be auto, python or rust"):
        StreamingSpikeCodec(backend="unsupported")
    with pytest.raises(ValueError, match="shape"):
        codec.compress(np.zeros(3, dtype=np.int8))
    with pytest.raises(ValueError, match="dimensions exceed"):
        codec.compress_frame(np.empty((65536, 0), dtype=np.int8))
    with pytest.raises(ValueError, match="dimensions exceed"):
        codec.compress(np.empty((0, 65536), dtype=np.int8))
    with pytest.raises(ValueError, match="only zeros and ones"):
        codec.compress_frame(np.array([[256]], dtype=np.int64))


@pytest.mark.parametrize("error", ["ImportError", "RuntimeError"])
def test_reports_denied_native_raster_loading(error: str) -> None:
    """Exercise the real loader under a host import restriction."""
    require_engine()
    assert streaming_codec.__file__ is not None
    expected = Path(streaming_codec.__file__).resolve()
    code = """
import builtins
from pathlib import Path
import sys

expected = Path(sys.argv[1])
sys.path.insert(0, str(expected.parents[2]))
exception = getattr(builtins, sys.argv[2])
attempts = []

def deny_extension(event, arguments):
    if event == "import" and arguments[0] == "sc_neurocore_engine.sc_neurocore_engine":
        attempts.append(arguments[0])
        raise exception("host denied raster extension loading")

sys.addaudithook(deny_extension)
import numpy as np
from sc_neurocore.spike_codec import StreamingSpikeCodec, probe_codec, streaming_codec

assert Path(streaming_codec.__file__).resolve() == expected
status = probe_codec("streaming")
assert status.backend_choices == ("python",)
assert attempts
assert any(exception.__name__ in item for item in status.diagnostics)
try:
    StreamingSpikeCodec()
except exception:
    pass
else:
    raise AssertionError("automatic selection silently hid a broken native installation")
python = StreamingSpikeCodec(backend="python")
raster = np.array([[1, 0], [0, 1]], dtype=np.int8)
np.testing.assert_array_equal(python.decompress(python.compress(raster)[0]), raster)
"""
    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(expected), error],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
