# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — executable EscapeRate polyglot parity

"""Seeded event, voltage, RNG, and distribution contracts for all runtimes."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import escape_rate as backends
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_RUNNERS = {
    "rust": backends.simulate_rust,
    "julia": backends.simulate_julia,
    "go": backends.simulate_go,
    "mojo": backends.simulate_mojo,
}


def _backend_ready(backend: str) -> bool:
    """Load one native runtime without relying on test execution order."""
    if backend == "rust":
        return backends._HAS_RUST
    return {
        "julia": backends.ensure_julia_loaded,
        "go": backends.ensure_go_loaded,
        "mojo": backends.ensure_mojo_loaded,
    }[backend]()


@pytest.mark.parametrize(
    ("events", "final_rng", "message"),
    [
        (np.array([0.0, 1.5]), 1, "non-binary events"),
        (np.array([0.0, 256.0]), 1, "non-binary events"),
        (np.array([0.0, 1.0]), 1.5, "invalid LFSR state"),
    ],
)
def test_backend_normalisation_rejects_values_that_casts_would_truncate(
    events: npt.NDArray[np.float64],
    final_rng: float,
    message: str,
) -> None:
    """Validate raw numeric values before narrowing to uint8 or integer state."""
    with pytest.raises(FloatingPointError, match=message):
        backends._normalise_result(np.array([-70.0, -69.0]), events, -69.0, final_rng)


def _configured() -> EscapeRateNeuron:
    """Exercise every physical and stochastic ABI field."""
    return EscapeRateNeuron(
        v=-64.0,
        v_rest=-68.0,
        v_reset=-66.0,
        v_threshold=-52.0,
        tau_m=12.5,
        rho_0=0.02,
        delta_u=4.0,
        resistance=1.3,
        dt=0.25,
        seed=0x1234,
    )


def _arguments(
    neuron: EscapeRateNeuron, n_steps: int, current: float
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    int,
    float,
]:
    return (
        neuron.v,
        neuron.v_rest,
        neuron.v_reset,
        neuron.v_threshold,
        neuron.tau_m,
        neuron.rho_0,
        neuron.delta_u,
        neuron.resistance,
        neuron.dt,
        neuron.rng_state,
        n_steps,
        current,
    )


def _reference(
    n_steps: int = 4096, current: float = 17.0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], float, int]:
    neuron = _configured()
    return neuron._simulate_python(n_steps, current)


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four claimed native runtimes without a skipped surrogate."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_full_contract_backends_match_seeded_python(backend: str) -> None:
    """Preserve every event, the voltage trace, and the final LFSR state."""
    assert _backend_ready(backend)
    expected_trace, expected_events, expected_v, expected_rng = _reference()
    neuron = _configured()
    trace, events, final_v, final_rng = _RUNNERS[backend](*_arguments(neuron, 4096, 17.0))

    np.testing.assert_array_equal(events, expected_events)
    np.testing.assert_allclose(trace, expected_trace, rtol=0.0, atol=2.0e-14)
    assert final_v == pytest.approx(expected_v, rel=0.0, abs=2.0e-14)
    assert final_rng == expected_rng
    assert int(events.sum()) == 29


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_voltage_and_rng(backend: str) -> None:
    """A zero-step batch has no hidden RNG advance or default-state injection."""
    neuron = _configured()
    before = (neuron.v, neuron.rng_state)
    trace, spikes = neuron.simulate(0, current=17.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert (neuron.v, neuron.rng_state) == before


def test_public_reset_replays_voltage_events_and_rng() -> None:
    """Reset restores both state dimensions required for exact reproducibility."""
    neuron = _configured()
    neuron.reset()
    first_trace, first_spikes = neuron.simulate(2048, current=17.0, backend="python")
    first_rng = neuron.rng_state
    neuron.reset()
    replay_trace, replay_spikes = neuron.simulate(2048, current=17.0, backend="python")
    np.testing.assert_array_equal(replay_trace, first_trace)
    assert replay_spikes == first_spikes
    assert neuron.rng_state == first_rng


def test_auto_dispatch_matches_the_rust_batch() -> None:
    """The current measured low-overhead native default is the Rust batch."""
    automatic = _configured()
    expected = _configured()
    auto_trace, auto_spikes = automatic.simulate(4096, current=17.0, backend="auto")
    rust_trace, rust_spikes = expected.simulate(4096, current=17.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, rust_trace)
    assert auto_spikes == rust_spikes
    assert (automatic.v, automatic.rng_state) == (expected.v, expected.rng_state)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_full_lfsr_period_has_exact_rate_and_geometric_isi_class(backend: str) -> None:
    """One complete decimated period proves rate and ISI-distribution fidelity."""
    assert _backend_ready(backend)
    rho_quarter = -math.log(0.75)
    neuron = EscapeRateNeuron(
        v=-50.0,
        v_rest=-50.0,
        v_reset=-50.0,
        v_threshold=-50.0,
        rho_0=rho_quarter,
        dt=1.0,
        seed=0xACE1,
    )
    expected = neuron._simulate_python(0xFFFF, 0.0)
    probe = EscapeRateNeuron(
        v=-50.0,
        v_rest=-50.0,
        v_reset=-50.0,
        v_threshold=-50.0,
        rho_0=rho_quarter,
        dt=1.0,
        seed=0xACE1,
    )
    actual = _RUNNERS[backend](*_arguments(probe, 0xFFFF, 0.0))
    np.testing.assert_array_equal(actual[1], expected[1])
    assert int(actual[1].sum()) == 16_383
    assert actual[3] == expected[3] == 0xACE1
    intervals = np.diff(np.flatnonzero(actual[1]))
    assert float(intervals.mean()) == pytest.approx(4.0, abs=5.0e-4)
    assert float(intervals.std() / intervals.mean()) == pytest.approx(math.sqrt(0.75), abs=0.01)


def test_rust_safety_module_matches_the_public_python_stream(tmp_path: Path) -> None:
    """Compile the separate Rust-safety recurrence and compare all live state."""
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/escape_rate.rs"
    program = tmp_path / "escape_rate_trace.rs"
    binary = tmp_path / "escape_rate_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = EscapeRateNeuron {{
        v: -64.0, v_rest: -68.0, v_reset: -66.0, v_threshold: -52.0,
        tau_m: 12.5, rho_0: 0.02, delta_u: 4.0, resistance: 1.3, dt: 0.25,
        rng_state: 0x1234, initial_seed: 0x1234,
    }};
    for _ in 0..4096 {{
        let event = state.step(17.0).expect("valid EscapeRate step");
        println!("ESCAPE_TRACE {{}} {{:.17}} {{}}", event, state.v, state.rng_state);
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
    rust_events = np.asarray([int(row[1]) for row in rows], dtype=np.uint8)
    rust_trace = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    rust_rng = int(rows[-1][3])
    expected_trace, expected_events, _, expected_rng = _reference()
    np.testing.assert_array_equal(rust_events, expected_events)
    np.testing.assert_allclose(rust_trace, expected_trace, rtol=0.0, atol=2.0e-14)
    assert rust_rng == expected_rng
