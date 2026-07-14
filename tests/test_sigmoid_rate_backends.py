# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable sigmoid-rate five-backend parity

"""Full parameters, traces, errors, and atomic state across every lane."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import sigmoid_rate as backends
from sc_neurocore.neurons.models.sigmoid_rate import SigmoidRateNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_CONFIG = {"r": 0.25, "tau": 10.0, "beta": 2.0, "theta": 1.0, "dt": 0.5}


@pytest.mark.parametrize(
    ("trace", "final_rate", "message"),
    (
        ([[0.25]], 0.25, "malformed rate trace"),
        ([float("nan")], 0.25, "non-finite or unbounded"),
        ([1.1], 1.1, "non-finite or unbounded"),
        ([0.25], float("nan"), "invalid final rate"),
        ([0.25], 0.5, "disagrees with its trace"),
    ),
)
def test_normaliser_rejects_malformed_results(
    trace: object,
    final_rate: object,
    message: str,
) -> None:
    """Native output is validated before narrowing or state commit."""
    with pytest.raises(FloatingPointError, match=message):
        backends.normalise_result(
            cast(npt.ArrayLike, trace),
            final_rate,
            n_steps=1,
            initial_rate=0.25,
        )


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four compiled runtimes without skipped surrogates."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_complete_configured_contract_matches_python(backend: str) -> None:
    """Every lane preserves the complete parameterized rate trajectory."""
    reference = SigmoidRateNeuron(**_CONFIG)
    expected = reference.simulate(512, 3.0, backend="python")
    actual_neuron = SigmoidRateNeuron(**_CONFIG)
    actual = actual_neuron.simulate(512, 3.0, backend=backend)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-12)
    assert actual_neuron.r == pytest.approx(reference.r, rel=0.0, abs=5.0e-12)


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_initial_rate(backend: str) -> None:
    """A zero-step batch does not inject defaults or phantom activity."""
    neuron = SigmoidRateNeuron(**_CONFIG)
    trace = neuron.simulate(0, 3.0, backend=backend)
    assert trace.shape == (0,)
    assert neuron.r == _CONFIG["r"]


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_large_timestep_remains_bounded(backend: str) -> None:
    """Exact relaxation remains inside the unit interval when dt exceeds tau."""
    neuron = SigmoidRateNeuron(r=1.0, tau=0.1, beta=1.0, theta=0.0, dt=5.0)
    trace = neuron.simulate(2, -100.0, backend=backend)
    assert np.logical_and(trace >= 0.0, trace <= 1.0).all()
    assert trace[-1] < 1.0e-12


def test_auto_uses_first_available_measured_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto consults the selector once and propagates the chosen runner."""
    calls: list[str] = []
    monkeypatch.setattr(
        backends,
        "select_backend_order",
        lambda _kernel, static: ("go", "rust", "mojo", "julia", "python"),
    )
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: True)

    def fake_go(*args: float | int) -> backends.SigmoidRateResult:
        calls.append("go")
        n_steps = int(args[-2])
        initial = float(args[0])
        return np.full(n_steps, initial), initial

    monkeypatch.setattr(backends, "simulate_go", fake_go)
    neuron = SigmoidRateNeuron(r=0.25)
    np.testing.assert_array_equal(neuron.simulate(2, 3.0), np.full(2, 0.25))
    assert calls == ["go"]


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
    neuron = SigmoidRateNeuron(r=0.25)
    with pytest.raises(RuntimeError, match=backend.title()):
        neuron.simulate(1, 3.0, backend=backend)
    assert neuron.r == 0.25


class _RejectingCAbi:
    def sigmoid_rate_simulate_c(self, *_args: object) -> int:
        return -1


def test_public_c_runners_name_rejected_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both C lanes surface rejection with backend-specific diagnostics."""
    rejecting = _RejectingCAbi()
    monkeypatch.setattr(backends, "_go_lib", rejecting)
    monkeypatch.setattr(backends, "_mojo_lib", rejecting)
    args = (0.25, 10.0, 2.0, 1.0, 0.5, 1, 3.0)
    with pytest.raises(FloatingPointError, match="Go SigmoidRate kernel rejected"):
        backends.simulate_go(*args)
    with pytest.raises(FloatingPointError, match="Mojo SigmoidRate kernel rejected"):
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
    status = library.sigmoid_rate_simulate_c(
        0.25,
        0.0,
        2.0,
        1.0,
        0.5,
        1,
        3.0,
        destination,
    )
    assert status == -1
    np.testing.assert_array_equal(output, np.full(2, -999.0))


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    """Compile the separate safety recurrence and compare its configured trace."""
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/sigmoid_rate.rs"
    program = tmp_path / "sigmoid_rate_trace.rs"
    binary = tmp_path / "sigmoid_rate_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = SigmoidRateNeuron::with_parameters(0.25, 10.0, 2.0, 1.0, 0.5)
        .expect("valid sigmoid-rate contract");
    for _ in 0..6 {{
        println!("{{:.17}}", state.step(3.0).expect("valid sigmoid-rate step"));
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
    actual = np.asarray([float(value) for value in completed.stdout.splitlines()])
    reference = SigmoidRateNeuron(**_CONFIG).simulate(6, 3.0, backend="python")
    np.testing.assert_array_equal(actual, reference)
