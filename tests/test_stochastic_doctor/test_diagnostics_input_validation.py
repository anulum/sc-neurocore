# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic doctor input validation on both backends

"""Hold the stochastic doctor to bitstreams of zeros and ones on the Python and the native path.

Values other than 0 and 1 used to be counted by value on the Python path and by set bits in the native
core, divided by zero in the Python SCC and indexed past the histogram; a float or out-of-range integer
was truncated or wrapped to a byte first. An empty layer produced NaN fields and NaN tokens in its JSON.
Every public entry point now checks its input before choosing a backend, so both backends see the same
binary bytes and the native core is never reached with anything else.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import sc_neurocore.stochastic_doctor.diagnostics as diag


class _NativeMustNotRun:
    """Stand-in native core that fails the test if any kernel is reached."""

    def __getattr__(self, name: str) -> Any:
        """Fail on any kernel lookup, naming the kernel."""
        raise AssertionError(f"native kernel {name} reached with invalid input")


class _NativeRecorder:
    """Stand-in native core that records what each kernel received."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.received: list[np.ndarray[Any, Any]] = []

    def py_scc_bytes(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        """Record both streams and return a fixed SCC."""
        self.received += [a, b]
        return 0.25

    def py_precision_bytes(self, bitstream: np.ndarray[Any, Any]) -> tuple[float, float]:
        """Record the stream and return a fixed estimate."""
        self.received.append(bitstream)
        return (0.5, 0.01)

    def py_histogram(self, bitstream: np.ndarray[Any, Any], word_size: int) -> list[int]:
        """Record the stream and return an empty histogram of ``word_size + 1`` bins."""
        self.received.append(bitstream)
        return [0] * (word_size + 1)


def _select(monkeypatch: pytest.MonkeyPatch, core: object | None) -> None:
    """Select the native path with ``core``, or the Python path when ``core`` is None."""
    monkeypatch.setattr(diag, "_HAS_PYO3", core is not None)
    monkeypatch.setattr(diag, "_sdc_rust", core)


@pytest.fixture(params=["python", "native"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run a test once on the Python path and once with a native core that must not be reached."""
    _select(monkeypatch, _NativeMustNotRun() if request.param == "native" else None)
    return str(request.param)


BINARY = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
NOT_BINARY: list[Any] = [
    np.array([2, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8),
    np.array([255, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8),
    np.array([0.5, 0, 1, 1, 0, 0, 1, 0]),
    np.array([-1, 0, 1, 1, 0, 0, 1, 0]),
    np.array([256, 0, 1, 1, 0, 0, 1, 0]),
    np.array([math.nan, 0, 1, 1, 0, 0, 1, 0]),
    np.array(["1", "0", "1", "1", "0", "0", "1", "0"]),
    np.array([1 + 0j, 0, 1, 1, 0, 0, 1, 0]),
]


@pytest.mark.parametrize("stream", NOT_BINARY, ids=lambda s: str(s[0]))
def test_compute_scc_refuses_values_other_than_zero_and_one(backend: str, stream: Any) -> None:
    """A 2, 255, 0.5, -1, 256, NaN, text or complex entry is refused on either backend, as either argument."""
    with pytest.raises(ValueError, match="0 and 1"):
        diag.compute_scc(stream, BINARY)
    with pytest.raises(ValueError, match="0 and 1"):
        diag.compute_scc(BINARY, stream)


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        (np.array([], dtype=np.uint8), np.array([], dtype=np.uint8), "non-empty"),
        (BINARY, BINARY[:4], "same length"),
        (np.stack([BINARY, BINARY]), np.stack([BINARY, BINARY]), "one-dimensional"),
    ],
)
def test_compute_scc_refuses_empty_unequal_and_matrix_streams(
    backend: str, a: Any, b: Any, message: str
) -> None:
    """SCC needs two one-dimensional, non-empty streams of the same length."""
    with pytest.raises(ValueError, match=message):
        diag.compute_scc(a, b)


@pytest.mark.parametrize("stream", NOT_BINARY, ids=lambda s: str(s[0]))
def test_precision_and_histogram_refuse_values_other_than_zero_and_one(
    backend: str, stream: Any
) -> None:
    """The precision estimate and the popcount histogram refuse the same inputs as SCC."""
    doctor = diag.StochasticDoctor()
    with pytest.raises(ValueError, match="0 and 1"):
        doctor.estimate_precision(stream)
    with pytest.raises(ValueError, match="0 and 1"):
        doctor.compute_histogram(stream, word_size=4)


@pytest.mark.parametrize("word_size", [0, -4, True, 2.0])
def test_the_histogram_word_size_is_a_positive_integer(backend: str, word_size: Any) -> None:
    """A zero, negative, boolean or float word size is refused before any backend runs."""
    with pytest.raises(ValueError, match="word_size"):
        diag.StochasticDoctor().compute_histogram(BINARY, word_size=word_size)


@pytest.mark.parametrize(
    "streams",
    [np.zeros((0, 8), dtype=np.uint8), np.zeros((2, 0), dtype=np.uint8), BINARY],
    ids=["no-neurons", "no-bits", "one-dimensional"],
)
def test_audit_layer_refuses_empty_or_flat_layers(backend: str, streams: Any) -> None:
    """A layer audit needs a (neurons, bits) array with at least one neuron and one bit."""
    with pytest.raises(ValueError, match="bitstreams"):
        diag.StochasticDoctor().audit_layer("empty", streams)


def test_the_native_core_receives_contiguous_binary_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Booleans, Python lists and int64 zeros and ones reach the native core as uint8 bytes."""
    recorder = _NativeRecorder()
    _select(monkeypatch, recorder)
    doctor = diag.StochasticDoctor()
    assert diag.compute_scc(BINARY.astype(bool), [1, 0, 0, 1, 0, 1, 1, 0]) == 0.25
    assert doctor.estimate_precision(BINARY.astype(np.int64)) == (0.5, 0.01)
    assert doctor.compute_histogram(BINARY, word_size=4).tolist() == [0] * 5
    assert len(recorder.received) == 4
    for stream in recorder.received:
        assert stream.dtype == np.uint8
        assert stream.flags.c_contiguous
        assert set(np.unique(stream).tolist()) <= {0, 1}


@pytest.mark.parametrize(
    "layer",
    [
        np.asfortranarray(np.stack([BINARY, BINARY[::-1]])),
        np.stack([BINARY, BINARY[::-1]]).T.copy().T,
    ],
    ids=["fortran-order", "transposed-view"],
)
def test_audit_layer_accepts_any_memory_order_on_both_backends(
    monkeypatch: pytest.MonkeyPatch, layer: np.ndarray[Any, Any]
) -> None:
    """A Fortran-ordered or transposed layer is audited alike: the native core gets contiguous rows."""
    recorder = _NativeRecorder()
    _select(monkeypatch, recorder)
    native = diag.StochasticDoctor().audit_layer("order", layer)
    _select(monkeypatch, None)
    python = diag.StochasticDoctor().audit_layer("order", layer)
    assert (
        (native.num_neurons, native.stream_length)
        == (python.num_neurons, python.stream_length)
        == (2, 8)
    )
    assert len(recorder.received) == 4
    assert all(stream.flags.c_contiguous for stream in recorder.received)


def test_the_native_layout_check_still_comes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-contiguous view is refused for its layout before its values are read."""
    _select(monkeypatch, _NativeMustNotRun())
    view = np.array([1, 9, 0, 9, 1, 9, 1, 9], dtype=np.uint8)[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        diag.compute_scc(view, np.ascontiguousarray(view))


def test_python_results_are_unchanged_for_binary_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid binary input gives the same SCC, precision and histogram as before."""
    _select(monkeypatch, None)
    doctor = diag.StochasticDoctor()
    other = np.array([1, 1, 0, 1, 0, 1, 1, 0], dtype=np.uint8)
    assert diag.compute_scc(BINARY, other) == diag._scc_python(BINARY, other)
    assert doctor.estimate_precision(BINARY) == (0.5, 0.5 * 0.5 / 8)
    assert doctor.estimate_precision(np.array([], dtype=np.uint8)) == (0.0, 0.0)
    assert doctor.compute_histogram(BINARY, word_size=4).tolist() == [0, 1, 0, 1, 0]


def test_report_json_never_carries_nan_tokens() -> None:
    """A report holding a NaN cannot be serialised: JSON per RFC 8259 has no NaN."""
    report = diag.BitstreamAuditReport(layer="x", stream_length=8, num_neurons=1)
    report.mean_precision = math.nan
    with pytest.raises(ValueError, match="JSON"):
        report.to_json()
