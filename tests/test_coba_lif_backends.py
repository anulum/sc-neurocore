# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — executable COBA LIF polyglot backend parity

"""End-to-end parity and rejection contracts for every native COBA LIF lane."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import coba_lif as backends
from sc_neurocore.neurons.models.coba_lif import COBALIFNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_STATE_ATOL = 1.0e-13


def _configured(*, refractory_time: float = 0.0) -> COBALIFNeuron:
    """Return a non-default cell exercising every field in the native ABI."""
    return COBALIFNeuron(
        v=-59.0,
        g_e=1.25,
        g_i=0.75,
        refractory_time=refractory_time,
        c_m=190.0,
        g_l=11.0,
        e_l=-61.0,
        e_e=2.0,
        e_i=-82.0,
        tau_e=4.5,
        tau_i=11.0,
        v_threshold=-50.5,
        v_reset=-62.0,
        refractory_period=3.7,
        dt=0.1,
    )


def _run(
    backend: str,
    *,
    n_steps: int = 400,
    current: float = 650.0,
    delta_ge: float = 0.15,
    delta_gi: float = 0.07,
    refractory_time: float = 0.0,
) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
    """Execute one public backend and return its trace, events, and final state."""
    neuron = _configured(refractory_time=refractory_time)
    trace, spikes = neuron.simulate(
        n_steps,
        current=current,
        delta_ge=delta_ge,
        delta_gi=delta_gi,
        backend=backend,
    )
    return trace, spikes, (neuron.v, neuron.g_e, neuron.g_i, neuron.refractory_time)


def _assert_parity(actual: npt.ArrayLike, expected: npt.ArrayLike) -> None:
    """Enforce the measured cross-runtime floating-point envelope."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=_STATE_ATOL,
    )


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four native runtimes without a skipped claimed surface."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


def test_auto_dispatch_uses_measured_fastest_julia_batch() -> None:
    """Route the public auto lane through the fastest controlled benchmark result."""
    automatic = _configured()
    expected = _configured()

    auto_trace, auto_spikes = automatic.simulate(
        400,
        current=650.0,
        delta_ge=0.15,
        delta_gi=0.07,
        backend="auto",
    )
    expected_trace, expected_spikes = expected.simulate(
        400,
        current=650.0,
        delta_ge=0.15,
        delta_gi=0.07,
        backend="julia",
    )

    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert auto_spikes == expected_spikes
    assert (automatic.v, automatic.g_e, automatic.g_i, automatic.refractory_time) == (
        expected.v,
        expected.g_e,
        expected.g_i,
        expected.refractory_time,
    )


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_full_contract_backends_match_conductance_driven_python(backend: str) -> None:
    """Preserve the trace, spikes, refractory holds, and all four final states."""
    reference_trace, reference_spikes, reference_state = _run("python")
    trace, spikes, state = _run(backend)

    assert reference_spikes == 6
    assert spikes == reference_spikes
    _assert_parity(trace, reference_trace)
    _assert_parity(state, reference_state)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_backends_match_when_the_initial_state_is_refractory(backend: str) -> None:
    """Exercise the exact timer clamp before integration resumes."""
    reference_trace, reference_spikes, reference_state = _run(
        "python",
        n_steps=5,
        current=0.0,
        delta_ge=0.0,
        delta_gi=0.0,
        refractory_time=0.3,
    )
    trace, spikes, state = _run(
        backend,
        n_steps=5,
        current=0.0,
        delta_ge=0.0,
        delta_gi=0.0,
        refractory_time=0.3,
    )

    assert reference_trace[:3].tolist() == [-62.0, -62.0, -62.0]
    assert reference_spikes == spikes == 0
    _assert_parity(trace, reference_trace)
    _assert_parity(state, reference_state)


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_run_preserves_the_complete_state(backend: str) -> None:
    """A zero-step batch returns no samples and commits no hidden defaults."""
    neuron = _configured(refractory_time=0.2)
    before = (neuron.v, neuron.g_e, neuron.g_i, neuron.refractory_time)

    trace, spikes = neuron.simulate(0, current=650.0, backend=backend)

    assert trace.shape == (0,)
    assert spikes == 0
    assert (neuron.v, neuron.g_e, neuron.g_i, neuron.refractory_time) == before


@pytest.mark.parametrize("backend", ("rust", "go", "mojo"))
def test_compiled_candidate_rejection_is_atomic(backend: str) -> None:
    """Compiled kernels validate the raw RK4 voltage before threshold reset."""
    neuron = COBALIFNeuron(v=90.0)
    before = (neuron.v, neuron.g_e, neuron.g_i, neuron.refractory_time)
    expected_error = ValueError if backend == "rust" else FloatingPointError

    with pytest.raises(expected_error):
        neuron.simulate(1, current=1.0e8, backend=backend)

    assert (neuron.v, neuron.g_e, neuron.g_i, neuron.refractory_time) == before


def test_rust_safety_module_matches_the_public_python_trace(tmp_path: Path) -> None:
    """Compile and execute the separate Rust-safety recurrence on all ABI fields."""
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/coba_lif.rs"
    program = tmp_path / "coba_lif_trace.rs"
    binary = tmp_path / "coba_lif_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = COBALIFNeuron {{
        v: -59.0,
        g_e: 1.25,
        g_i: 0.75,
        refractory_time: 0.0,
        c_m: 190.0,
        g_l: 11.0,
        e_l: -61.0,
        e_e: 2.0,
        e_i: -82.0,
        tau_e: 4.5,
        tau_i: 11.0,
        v_threshold: -50.5,
        v_reset: -62.0,
        refractory_period: 3.7,
        dt: 0.1,
    }};
    for _ in 0..400 {{
        let spike = state
            .step_with_conductance(650.0, 0.15, 0.07)
            .expect("valid COBA LIF recurrence");
        println!(
            "COBA_TRACE {{}} {{:.17}} {{:.17}} {{:.17}} {{:.17}}",
            spike, state.v, state.g_e, state.g_i, state.refractory_time
        );
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
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = [line.split() for line in completed.stdout.splitlines()]
    rust_events = [int(row[1]) for row in rows]
    rust_trace = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    rust_state = tuple(float(value) for value in rows[-1][2:6])
    python_trace, python_spikes, python_state = _run("python")

    assert len(rows) == 400
    assert sum(rust_events) == python_spikes == 6
    _assert_parity(rust_trace, python_trace)
    _assert_parity(rust_state, python_state)
