# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable McCulloch-Pitts five-backend parity

"""Exact truth rows, strict failures and atomic output across every lane."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import mcculloch_pitts as backends
from sc_neurocore.neurons.models.mcculloch_pitts import McCullochPittsNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_COUNTS = np.asarray([0, 1, 2, 3, (1 << 31) - 1], dtype=np.int64)
_FLAGS = np.asarray([False, False, False, True, False], dtype=np.bool_)


@pytest.mark.parametrize(
    ("result", "length", "message"),
    (
        (([[0]], 0), 1, "malformed event trace"),
        (([0, 2], 1), 2, "non-binary events"),
        (([0, float("nan")], 0), 2, "non-binary events"),
        (([0], 1.5), 1, "invalid event count"),
        (([0], True), 1, "invalid event count"),
        (([0, 1], 0), 2, "disagrees with its event trace"),
        (([0], 0, 0), 1, "malformed result"),
    ),
)
def test_normaliser_rejects_malformed_or_lossy_results(
    result: object,
    length: int,
    message: str,
) -> None:
    """Raw native results are checked before uint8 narrowing."""
    with pytest.raises(FloatingPointError, match=message):
        backends.normalise_result(result, expected_length=length)


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four real compiled runtimes without skipped surrogates."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_source_truth_rows_are_bit_exact_on_every_backend(backend: str) -> None:
    """All compiled lanes preserve OR, threshold equality and absolute veto."""
    expected = McCullochPittsNeuron(theta=2).simulate(
        _COUNTS,
        _FLAGS,
        backend="python",
    )
    actual = McCullochPittsNeuron(theta=2).simulate(_COUNTS, _FLAGS, backend=backend)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1] == 2


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_is_exact_and_stateless(backend: str) -> None:
    """A zero-row batch neither invents state nor emits a phantom event."""
    events, event_count = McCullochPittsNeuron(theta=7).simulate([], [], backend=backend)
    assert events.shape == (0,)
    assert events.dtype == np.uint8
    assert event_count == 0


def test_auto_uses_first_available_measured_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto consults the selector once and executes only its chosen runner."""
    calls: list[str] = []
    monkeypatch.setattr(
        backends,
        "select_backend_order",
        lambda _kernel, static: ("go", "rust", "mojo", "julia", "python"),
    )
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: True)

    def fake_go(
        theta: int,
        counts: np.ndarray[Any, np.dtype[np.int64]],
        flags: np.ndarray[Any, np.dtype[np.uint8]],
    ) -> backends.McCullochPittsResult:
        calls.append("go")
        assert theta == 2
        events = np.asarray((flags == 0) & (counts >= theta), dtype=np.uint8)
        return events, int(events.sum())

    monkeypatch.setattr(backends, "evaluate_go", fake_go)
    events, event_count = McCullochPittsNeuron(theta=2).simulate([1, 2], backend="auto")
    assert calls == ["go"]
    assert events.tolist() == [0, 1]
    assert event_count == 1


def test_auto_does_not_fall_through_an_available_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only availability falls through; selected execution failures propagate."""
    monkeypatch.setattr(
        backends,
        "select_backend_order",
        lambda _kernel, static: ("go", "python"),
    )
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: True)

    def broken(*_args: object) -> backends.McCullochPittsResult:
        raise FloatingPointError("native failure")

    monkeypatch.setattr(backends, "evaluate_go", broken)
    with pytest.raises(FloatingPointError, match="native failure"):
        McCullochPittsNeuron().simulate([1], backend="auto")


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit requests fail closed instead of silently using Python."""
    if backend == "rust":
        monkeypatch.setattr(backends, "_HAS_RUST", False)
        monkeypatch.setattr(backends, "_engine_evaluate", None)
    else:
        monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        McCullochPittsNeuron().simulate([1], backend=backend)


class _RejectingCAbi:
    def mcculloch_pitts_evaluate_c(self, *_args: object) -> int:
        return -1


def test_public_c_runners_name_rejected_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both C lanes surface rejection with backend-specific diagnostics."""
    rejecting = _RejectingCAbi()
    monkeypatch.setattr(backends, "_go_lib", rejecting)
    monkeypatch.setattr(backends, "_mojo_lib", rejecting)
    counts = np.asarray([1], dtype=np.int64)
    flags = np.asarray([0], dtype=np.uint8)
    with pytest.raises(FloatingPointError, match="Go McCulloch-Pitts kernel rejected"):
        backends.evaluate_go(1, counts, flags)
    with pytest.raises(FloatingPointError, match="Mojo McCulloch-Pitts kernel rejected"):
        backends.evaluate_mojo(1, counts, flags)


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_invalid_contract_writes_nothing(backend: str) -> None:
    """Go staging and Mojo validation passes make rejected writes atomic."""
    assert {"go": backends.ensure_go_loaded, "mojo": backends.ensure_mojo_loaded}[backend]()
    library: Any = backends._go_lib if backend == "go" else backends._mojo_lib
    counts = np.asarray([-1, 2], dtype=np.int64)
    flags = np.asarray([0, 0], dtype=np.uint8)
    output = np.full(2, 77, dtype=np.uint8)
    if backend == "go":
        result = library.mcculloch_pitts_evaluate_c(
            1,
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(counts),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    else:
        result = library.mcculloch_pitts_evaluate_c(
            1,
            int(counts.ctypes.data),
            int(flags.ctypes.data),
            len(counts),
            int(output.ctypes.data),
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(2, 77, dtype=np.uint8))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_accepts_zero_rows_without_buffers(backend: str) -> None:
    """Both C contracts define an empty stateless batch without dummy memory."""
    assert {"go": backends.ensure_go_loaded, "mojo": backends.ensure_mojo_loaded}[backend]()
    library: Any = backends._go_lib if backend == "go" else backends._mojo_lib
    if backend == "go":
        result = library.mcculloch_pitts_evaluate_c(1, None, None, 0, None)
    else:
        result = library.mcculloch_pitts_evaluate_c(1, 0, 0, 0, 0)
    assert result == 0


def test_standalone_rust_safety_matches_public_truth_rows(tmp_path: Path) -> None:
    """Compile the separate safety kernel and compare its complete batch."""
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/mcculloch_pitts.rs"
    program = tmp_path / "mcculloch_pitts_trace.rs"
    binary = tmp_path / "mcculloch_pitts_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let neuron = McCullochPittsNeuron::new(2).expect("valid threshold");
    let (events, count) = neuron
        .evaluate_batch(&[0, 1, 2, 3, 2147483647], &[0, 0, 0, 1, 0])
        .expect("valid source rows");
    println!("{{:?}} {{}}", events, count);
}}
''',
        encoding="utf-8",
    )
    subprocess.run(
        ["rustc", "--edition", "2021", "-O", str(program), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    completed = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    expected_events, expected_count = McCullochPittsNeuron(theta=2).simulate(
        _COUNTS,
        _FLAGS,
        backend="python",
    )
    assert completed.stdout.strip() == f"{expected_events.tolist()} {expected_count}"
