# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — executable Poisson polyglot parity

"""Seeded event, RNG-state, rate, and interval contracts for all runtimes."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import poisson as backends
from sc_neurocore.neurons.models.poisson import PoissonNeuron

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


def _configured() -> PoissonNeuron:
    """Exercise every physical and stochastic ABI field."""
    return PoissonNeuron(rate_hz=250.0, dt_ms=1.0, seed=0x1234)


def _reference(
    n_steps: int = 4096, rate_override: float = -1.0
) -> tuple[npt.NDArray[np.uint8], int]:
    neuron = _configured()
    events, _count = neuron.simulate(n_steps, rate_override, backend="python")
    return events, neuron.rng_state


@pytest.mark.parametrize(
    ("events", "final_rng", "message"),
    [
        (np.array([[0.0, 1.0]]), 1, "malformed event trace"),
        (np.array([0.0, 1.5]), 1, "non-binary events"),
        (np.array([0.0, np.nan]), 1, "non-binary events"),
        (np.array([0.0, 1.0]), 0, "invalid LFSR state"),
        (np.array([0.0, 1.0]), 1.5, "invalid LFSR state"),
    ],
)
def test_backend_normalisation_rejects_malformed_or_lossy_values(
    events: npt.NDArray[np.float64],
    final_rng: float,
    message: str,
) -> None:
    """Validate raw numeric values before narrowing to uint8 or integer state."""
    with pytest.raises(FloatingPointError, match=message):
        backends._normalise_result(events, final_rng)


@pytest.mark.parametrize(
    ("events", "final_rng", "message"),
    [
        ([object()], 1, "non-numeric state"),
        ([0.0], object(), "non-numeric state"),
        ([0.0], True, "invalid LFSR state"),
        ([0.0], math.nan, "invalid LFSR state"),
        ([0.0], math.inf, "invalid LFSR state"),
        ([0.0], -1, "invalid LFSR state"),
        ([0.0], 0x1_0000, "invalid LFSR state"),
    ],
)
def test_backend_normalisation_rejects_non_numeric_or_out_of_range_rng(
    events: object,
    final_rng: object,
    message: str,
) -> None:
    """Reject values that cannot represent a binary trace and LFSR16 state."""
    with pytest.raises(FloatingPointError, match=message):
        backends._normalise_result(
            cast(npt.ArrayLike, events),
            cast(int | float, final_rng),
        )


def test_public_backend_runners_report_unavailable_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every explicit native runner has a deterministic unavailable error."""
    monkeypatch.setattr(backends, "_engine_simulate", None)
    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(backends, "_go_lib", None)
    monkeypatch.setattr(backends, "_mojo_lib", None)

    with pytest.raises(RuntimeError, match="Rust Poisson engine"):
        backends.simulate_rust(250.0, 1.0, 1, 1, -1.0)
    with pytest.raises(RuntimeError, match="Julia Poisson module"):
        backends.simulate_julia(250.0, 1.0, 1, 1, -1.0)
    with pytest.raises(RuntimeError, match="Go Poisson library"):
        backends.simulate_go(250.0, 1.0, 1, 1, -1.0)
    with pytest.raises(RuntimeError, match="Mojo Poisson library"):
        backends.simulate_mojo(250.0, 1.0, 1, 1, -1.0)


class _RejectingCAbi:
    def poisson_simulate_c(self, *_args: object) -> int:
        return -1


class _InconsistentGoCAbi:
    def poisson_simulate_c(
        self,
        _rate_hz: float,
        _dt_ms: float,
        _rng_state: int,
        _n_steps: int,
        _rate_override: float,
        destination: Any,
    ) -> int:
        destination[0] = 0.0
        destination[1] = 1.0
        return 1


def test_c_abi_rejection_names_go_and_mojo_and_preserves_fail_closed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both public C-ABI lanes surface a rejected native contract."""
    rejecting = _RejectingCAbi()
    monkeypatch.setattr(backends, "_go_lib", rejecting)
    monkeypatch.setattr(backends, "_mojo_lib", rejecting)

    with pytest.raises(FloatingPointError, match="Go Poisson kernel rejected"):
        backends.simulate_go(250.0, 1.0, 1, 1, -1.0)
    with pytest.raises(FloatingPointError, match="Mojo Poisson kernel rejected"):
        backends.simulate_mojo(250.0, 1.0, 1, 1, -1.0)


def test_c_abi_rejects_a_spike_count_that_disagrees_with_the_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scalar C result cannot contradict the binary event buffer."""
    monkeypatch.setattr(backends, "_go_lib", _InconsistentGoCAbi())
    with pytest.raises(FloatingPointError, match="spike count disagrees"):
        backends.simulate_go(250.0, 1.0, 1, 1, -1.0)


def test_every_acceleration_backend_is_executable() -> None:
    """Expose all four claimed native runtimes without a skipped surrogate."""
    assert backends._HAS_RUST
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_full_contract_backends_match_seeded_python(backend: str) -> None:
    """Preserve every event and the exact final LFSR state."""
    assert _backend_ready(backend)
    expected_events, expected_rng = _reference()
    events, final_rng = _RUNNERS[backend](250.0, 1.0, 0x1234, 4096, -1.0)

    np.testing.assert_array_equal(events, expected_events)
    assert final_rng == expected_rng == 45_999
    assert int(events.sum()) == 918


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_rng(backend: str) -> None:
    """A zero-step batch has no hidden RNG advance or default-state injection."""
    neuron = _configured()
    before = neuron.rng_state
    events, spikes = neuron.simulate(0, backend=backend)
    assert events.shape == (0,)
    assert spikes == 0
    assert neuron.rng_state == before


def test_public_reset_replays_events_and_rng() -> None:
    """Reset restores the execution state required for exact reproducibility."""
    neuron = _configured()
    first_events, first_spikes = neuron.simulate(2048, backend="python")
    first_rng = neuron.rng_state
    neuron.reset()
    replay_events, replay_spikes = neuron.simulate(2048, backend="python")
    np.testing.assert_array_equal(replay_events, first_events)
    assert replay_spikes == first_spikes
    assert neuron.rng_state == first_rng


def test_auto_dispatch_matches_the_integrated_rust_batch() -> None:
    """The production dispatcher selects the integrated Rust engine first."""
    automatic = _configured()
    expected = _configured()
    auto_events, auto_spikes = automatic.simulate(4096, backend="auto")
    rust_events, rust_spikes = expected.simulate(4096, backend="rust")
    np.testing.assert_array_equal(auto_events, rust_events)
    assert auto_spikes == rust_spikes
    assert automatic.rng_state == expected.rng_state


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_full_lfsr_period_has_exact_rate_and_geometric_interval_class(backend: str) -> None:
    """One complete decimated period proves rate and interval-distribution fidelity."""
    assert _backend_ready(backend)
    expected = PoissonNeuron(rate_hz=250.0, dt_ms=1.0, seed=0xACE1)
    expected_events, expected_count = expected.simulate(0xFFFF, backend="python")
    actual_events, actual_rng = _RUNNERS[backend](250.0, 1.0, 0xACE1, 0xFFFF, -1.0)
    np.testing.assert_array_equal(actual_events, expected_events)
    assert int(actual_events.sum()) == expected_count == 14_496
    assert actual_rng == expected.rng_state == 0xACE1
    probability = -math.expm1(-0.25)
    intervals = np.diff(np.flatnonzero(actual_events))
    assert float(intervals.mean()) == pytest.approx(1.0 / probability, abs=1.0e-3)
    assert float(intervals.std() / intervals.mean()) == pytest.approx(
        math.sqrt(1.0 - probability), abs=0.01
    )


def test_rust_safety_module_matches_the_public_python_stream(tmp_path: Path) -> None:
    """Compile the separate Rust-safety recurrence and compare all live state."""
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/poisson.rs"
    program = tmp_path / "poisson_trace.rs"
    binary = tmp_path / "poisson_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = PoissonNeuron {{
        rate_hz: 250.0, dt_ms: 1.0, rng_state: 0x1234, initial_seed: 0x1234,
    }};
    for _ in 0..4096 {{
        let event = state.step(-1.0).expect("valid Poisson step");
        println!("POISSON_TRACE {{}} {{}}", event, state.rng_state);
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
    rust_rng = int(rows[-1][2])
    expected_events, expected_rng = _reference()
    np.testing.assert_array_equal(rust_events, expected_events)
    assert rust_rng == expected_rng
