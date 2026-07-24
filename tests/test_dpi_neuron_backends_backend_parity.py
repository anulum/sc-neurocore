# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (backend_parity) from former test_dpi_neuron_backends.py

from __future__ import annotations

from tests.dpi_neuron_backends_support import *  # noqa: F403

def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four compiled lanes without a skipped parity surface."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize(("current", "expected_spikes"), _GOLDENS)
@pytest.mark.parametrize("backend", _FULL_CONTRACT_BACKENDS)
def test_full_contract_backends_match_python_factory_vector(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve events, three states, and physical-domain handling."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)
    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    _assert_state_parity(trace, reference_trace)
    _assert_state_parity(state, reference_state)


@pytest.mark.parametrize(("current", "expected_spikes"), _GOLDENS)
def test_factory_rust_matches_python(current: float, expected_spikes: int) -> None:
    """Prove the fixed-constructor PyO3 engine executes the same recurrence."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run("rust", current=current)
    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    _assert_state_parity(trace, reference_trace)
    _assert_state_parity(state, reference_state)


def test_rust_safety_executable_matches_configured_python_trace() -> None:
    """Run the separately maintained Rust-safety module on all 18 fields."""
    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        "src/sc_neurocore/accel/rust/Cargo.toml",
        "--example",
        "dpi_neuron_trace",
        "--",
        "0.37",
        "0.08",
        "0.0",
        "1.3",
        "0.2",
        "0.15",
        "0.9",
        "1.4",
        "0.12",
        "0.8",
        "4.2",
        "0.02",
        "0.65",
        "8.0",
        "7.0",
        "45.0",
        "0.6",
        "0.05",
        "400",
        "5.0",
    ]
    environment = dict(os.environ)
    environment["CARGO_TARGET_DIR"] = str(_REPOSITORY / "target")
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=True,
    )
    rows = [line.split() for line in completed.stdout.splitlines() if line.startswith("DPI_TRACE ")]
    assert len(rows) == 400
    rust_events = [int(row[1]) for row in rows]
    rust_states = np.asarray([[float(value) for value in row[2:5]] for row in rows])
    python_trace, python_spikes, python_state = _run(
        "python",
        current=5.0,
        n_steps=400,
        configured=True,
    )
    assert sum(rust_events) == python_spikes == 4
    _assert_state_parity(rust_states[:, 0], python_trace)
    _assert_state_parity(rust_states[-1], python_state)


@pytest.mark.parametrize("backend", _FULL_CONTRACT_BACKENDS)
def test_complete_configured_contract_matches_python(backend: str) -> None:
    """Carry every state and parameter field through the native ABI."""
    reference_trace, reference_spikes, reference_state = _run(
        "python", current=5.0, n_steps=400, configured=True
    )
    trace, spikes, state = _run(backend, current=5.0, n_steps=400, configured=True)
    assert spikes == reference_spikes == 4
    _assert_state_parity(trace, reference_trace)
    _assert_state_parity(state, reference_state)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_all_states(backend: str) -> None:
    """Return an empty trace without discarding any dynamic state."""
    neuron = DPINeuron() if backend == "rust" else _configured()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    trace, spikes = neuron.simulate(0, 5.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before
