# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects_and_flow) from former test_quadratic_if_backends.py

from __future__ import annotations

from tests.quadratic_if_backends_support import *  # noqa: F403


def test_rust_accepts_complete_non_default_contract() -> None:
    """Carry arbitrary valid parameters through the checked production batch."""
    neuron = _configured()
    reference = _configured()
    expected_voltage, expected_events = reference.simulate_complete(64, 2.2, "python")
    voltage, events = neuron.simulate_complete(64, 2.2, "rust")
    np.testing.assert_allclose(voltage, expected_voltage, rtol=0.0, atol=_TRACE_ATOL)
    np.testing.assert_array_equal(events, expected_events)
    assert neuron.v == pytest.approx(reference.v, rel=0.0, abs=_TRACE_ATOL)


def test_source_factory_preserves_latham_numerical_boundaries() -> None:
    """Do not conflate the unstable equilibrium with Latham's event apex."""
    source = QuadraticIFNeuron.latham_2000()
    assert source.v == -1.0
    assert source.v_reset == -3.0
    assert source.v_peak == pytest.approx(31.0 / 3.0)
    assert source.dt == 0.05
    assert source.profile == "latham_2000"


@pytest.mark.parametrize("backend", ("python", "rust", "julia", "go", "mojo"))
def test_source_complete_packet_matches_python(backend: str) -> None:
    """Require aligned per-step events for the count-bearing source profile."""
    if backend in ("julia", "go", "mojo"):
        _require_qif_backend(backend)
    reference = QuadraticIFNeuron.latham_2000()
    expected_voltage, expected_events = reference.simulate_complete(240, 4.0, "python")
    candidate = QuadraticIFNeuron.latham_2000()
    voltage, events = candidate.simulate_complete(240, 4.0, backend)
    np.testing.assert_allclose(voltage, expected_voltage, rtol=0.0, atol=_TRACE_ATOL)
    np.testing.assert_array_equal(events, expected_events)
    assert candidate.v == pytest.approx(reference.v, rel=0.0, abs=_TRACE_ATOL)


@pytest.mark.parametrize("n_steps", [-1, 1.0, True])
def test_invalid_step_count_fails_before_mutation(n_steps: object) -> None:
    """Reject negative and non-integer step counts at the public boundary."""
    neuron = QuadraticIFNeuron()
    before = neuron.v
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 0.0)
    assert neuron.v == before


def test_invalid_backend_fails_before_mutation() -> None:
    """Reject unknown dispatch selectors instead of silently using Python."""
    neuron = QuadraticIFNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 0.0, backend="cuda")
    assert neuron.v == -1.0


def test_non_finite_current_fails_before_mutation() -> None:
    """Apply the finite-input boundary to every dispatcher path."""
    neuron = QuadraticIFNeuron(v=-0.25)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan, backend="auto")
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.nan)
    assert neuron.v == -0.25


@pytest.mark.parametrize(
    "parameters",
    (
        {"v": math.nan},
        {"v": 1.0},
        {"v_reset": 1.0},
        {"dt": 0.0},
    ),
)
def test_invalid_model_contract_is_rejected_at_construction(
    parameters: dict[str, float],
) -> None:
    """Reject each invalid maintained field before a neuron can execute."""
    with pytest.raises(ValueError):
        QuadraticIFNeuron(**parameters)


def test_negative_current_exact_flow_covers_stable_and_reset_regimes() -> None:
    """Preserve the Riccati fixed point, regular decay, and finite-time reset."""
    fixed = QuadraticIFNeuron(v=-1.0)
    assert fixed.step(-1.0) == 0
    assert fixed.v == -1.0

    decaying = QuadraticIFNeuron(v=-0.25)
    assert decaying.step(-1.0) == 0
    assert math.isfinite(decaying.v)

    crossing = QuadraticIFNeuron(v=1.1, v_peak=3.0, dt=20.0)
    assert crossing.step(-1.0) == 1
    assert crossing.v == crossing.v_reset

    zero_current_crossing = QuadraticIFNeuron(v=0.5, v_peak=2.0, dt=3.0)
    assert zero_current_crossing.step(0.0) == 1
    assert zero_current_crossing.v == zero_current_crossing.v_reset


def test_exact_flow_rejects_overflow_without_mutation() -> None:
    """Translate a non-finite analytic candidate into mutation-free failure."""
    neuron = QuadraticIFNeuron(v=-0.25)
    before = neuron.v
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(-1.0e308)
    assert neuron.v == before


def test_reset_restores_only_the_runtime_state() -> None:
    """Restore voltage while retaining the configured parameters."""
    neuron = _configured()
    expected = (neuron.v_reset, neuron.v_peak, neuron.dt)
    neuron.step(2.2)
    neuron.reset()
    assert (neuron.v, neuron.v_peak, neuron.dt) == expected
