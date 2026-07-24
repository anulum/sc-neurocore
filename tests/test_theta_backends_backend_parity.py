# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (backend_parity) from former test_theta_backends.py

from __future__ import annotations

from tests.theta_backends_support import *  # noqa: F403


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four real compiled lanes without a skipped parity surface."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize(("current", "expected_spikes"), _GOLDENS)
@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_compiled_backends_match_python_golden(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve exact events and a tight circular cross-libm phase envelope."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)
    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    _assert_phase_parity(trace, reference_trace)
    _assert_phase_parity(state, reference_state)


def test_rust_safety_executable_matches_python_trace() -> None:
    """Run the actual accel/rust/safety Theta module against the Python trace."""
    command = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--manifest-path",
        "src/sc_neurocore/accel/rust/Cargo.toml",
        "--example",
        "theta_trace",
        "--",
        "0.0",
        "0.01",
        "400",
        "0.5",
    ]
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    rows = [
        line.split() for line in completed.stdout.splitlines() if line.startswith("THETA_TRACE ")
    ]
    assert len(rows) == 400
    rust_events = [int(row[1]) for row in rows]
    rust_trace = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    python_trace, python_spikes, _ = _run("python", current=0.5, n_steps=400)
    assert sum(rust_events) == python_spikes
    _assert_phase_parity(rust_trace, python_trace)


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_full_parameter_contract_matches_python(backend: str) -> None:
    """Carry phase and timestep across every full-parameter native ABI."""
    reference_trace, reference_spikes, reference_state = _run(
        "python", current=2.2, n_steps=400, factory=_configured
    )
    trace, spikes, state = _run(backend, current=2.2, n_steps=400, factory=_configured)
    assert reference_spikes > 0
    assert spikes == reference_spikes
    _assert_phase_parity(trace, reference_trace)
    _assert_phase_parity(state, reference_state)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding the initial phase."""
    neuron = ThetaNeuron() if backend == "rust" else _configured()
    before = neuron.theta
    trace, spikes = neuron.simulate(0, 2.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert neuron.theta == before
