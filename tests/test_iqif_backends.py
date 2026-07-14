# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable IQIF five-backend parity

"""Exact traces, full parameters, errors, and atomic state across every lane."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import iqif as backends
from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")


def _configured() -> IntegerQIFNeuron:
    return IntegerQIFNeuron(
        v=100,
        v_rest=96,
        v_threshold=180,
        v_reset=120,
        a=3,
        b=5,
        v_max=240,
        v_min=4,
    )


@pytest.mark.parametrize(
    ("trace", "spikes", "final_v", "message"),
    (
        ([[128.0]], 0, 128, "malformed voltage trace"),
        ([128.5], 0, 128, "non-integral or out-of-range"),
        ([float("nan")], 0, 128, "non-integral or out-of-range"),
        ([256.0], 0, 256, "non-integral or out-of-range"),
        ([128.0], 0, 129, "disagrees with its trace"),
        ([128.0], 1.5, 128, "invalid spike count"),
        ([128.0], True, 128, "invalid spike count"),
    ),
)
def test_normaliser_rejects_malformed_or_lossy_results(
    trace: object,
    spikes: object,
    final_v: object,
    message: str,
) -> None:
    """Raw native values are checked before int64 narrowing."""
    with pytest.raises(FloatingPointError, match=message):
        backends.normalise_result(
            cast(npt.ArrayLike, trace),
            spikes,
            final_v,
            n_steps=1,
            v_min=0,
            v_max=255,
        )


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four real compiled runtimes without skipped surrogates."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_source_tutorial_is_bit_exact_on_every_backend(backend: str) -> None:
    """All compiled lanes preserve every source state, event and final value."""
    expected = IntegerQIFNeuron()
    expected_trace, expected_spikes = expected.simulate(400, 10, backend="python")
    actual = IntegerQIFNeuron()
    trace, spikes = actual.simulate(400, 10, backend=backend)
    np.testing.assert_array_equal(trace, expected_trace)
    assert spikes == expected_spikes == 26
    assert actual.v == expected.v == 198


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_complete_configured_contract_matches_python(backend: str) -> None:
    """Every native ABI transports all state, slopes, boundaries and input."""
    expected = _configured()
    expected_trace, expected_spikes = expected.simulate(128, 17, backend="python")
    actual = _configured()
    trace, spikes = actual.simulate(128, 17, backend=backend)
    np.testing.assert_array_equal(trace, expected_trace)
    assert spikes == expected_spikes
    assert actual.v == expected.v


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_configured_state(backend: str) -> None:
    """A zero-step batch does not inject defaults or a phantom event."""
    neuron = _configured()
    trace, spikes = neuron.simulate(0, 17, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert neuron.v == 100


def test_auto_uses_first_available_measured_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto consults the selector once and propagates the chosen runner."""
    calls: list[str] = []
    monkeypatch.setattr(
        backends,
        "select_backend_order",
        lambda _kernel, static: ("go", "rust", "mojo", "julia", "python"),
    )
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: True)

    def fake_go(*args: int) -> backends.IQIFResult:
        calls.append("go")
        n_steps = args[-2]
        initial = args[0]
        return np.full(n_steps, initial, dtype=np.int64), 0, initial

    monkeypatch.setattr(backends, "simulate_go", fake_go)
    neuron = IntegerQIFNeuron()
    trace, spikes = neuron.simulate(2, 10, backend="auto")
    assert calls == ["go"]
    assert trace.tolist() == [128, 128]
    assert spikes == 0


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

    def broken(*_args: int) -> backends.IQIFResult:
        raise FloatingPointError("native failure")

    monkeypatch.setattr(backends, "simulate_go", broken)
    neuron = IntegerQIFNeuron()
    with pytest.raises(FloatingPointError, match="native failure"):
        neuron.simulate(1, 10, backend="auto")
    assert neuron.v == 128


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit requests fail closed rather than silently using Python."""
    if backend == "rust":
        monkeypatch.setattr(backends, "_HAS_RUST", False)
        monkeypatch.setattr(backends, "_engine_simulate", None)
    else:
        monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda: False)
    neuron = IntegerQIFNeuron()
    with pytest.raises(RuntimeError, match=backend.title()):
        neuron.simulate(1, 10, backend=backend)
    assert neuron.v == 128


class _RejectingCAbi:
    def iqif_simulate_c(self, *_args: object) -> int:
        return -1


def test_public_c_runners_name_rejected_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both C lanes surface rejection with backend-specific diagnostics."""
    rejecting = _RejectingCAbi()
    monkeypatch.setattr(backends, "_go_lib", rejecting)
    monkeypatch.setattr(backends, "_mojo_lib", rejecting)
    args = (128, 128, 200, 128, 1, 1, 255, 0, 1, 10)
    with pytest.raises(FloatingPointError, match="Go IQIF kernel rejected"):
        backends.simulate_go(*args)
    with pytest.raises(FloatingPointError, match="Mojo IQIF kernel rejected"):
        backends.simulate_mojo(*args)


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_invalid_contract_writes_nothing(backend: str) -> None:
    """Go staging and Mojo validation passes are caller-visible atomic."""
    assert {"go": backends.ensure_go_loaded, "mojo": backends.ensure_mojo_loaded}[backend]()
    library: Any = backends._go_lib if backend == "go" else backends._mojo_lib
    output = np.full(2, -999.0, dtype=np.float64)
    destination: Any = (
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if backend == "go"
        else int(output.ctypes.data)
    )
    result = library.iqif_simulate_c(
        128,
        128,
        200,
        128,
        -1,
        1,
        255,
        0,
        1,
        10,
        destination,
    )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(2, -999.0, dtype=np.float64))


def test_standalone_rust_safety_matches_public_trace(tmp_path: Path) -> None:
    """Compile the separate safety recurrence and compare all 400 rows."""
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/iqif.rs"
    program = tmp_path / "iqif_trace.rs"
    binary = tmp_path / "iqif_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = IntegerQIFNeuron::new();
    for _ in 0..400 {{
        let event = state.step(10).expect("valid IQIF step");
        println!("IQIF_TRACE {{}} {{}}", event, state.v);
    }}
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
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
    rows = [line.split() for line in completed.stdout.splitlines()]
    events = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
    trace = np.asarray([int(row[2]) for row in rows], dtype=np.int64)
    expected = IntegerQIFNeuron()
    expected_trace, expected_spikes = expected.simulate(400, 10, backend="python")
    np.testing.assert_array_equal(trace, expected_trace)
    assert int(events.sum()) == expected_spikes == 26
