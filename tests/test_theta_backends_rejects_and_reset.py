# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects_and_reset) from former test_theta_backends.py

from __future__ import annotations

from tests.theta_backends_support import *  # noqa: F403


def test_rust_accepts_non_default_complete_contract() -> None:
    """Carry configured phase and timestep through the checked PyO3 batch."""
    reference = _configured()
    neuron = _configured()
    expected_phase, expected_events = reference.simulate_complete(400, 2.2, "python")
    phase, events = neuron.simulate_complete(400, 2.2, "rust")
    _assert_phase_parity(phase, expected_phase)
    np.testing.assert_array_equal(events, expected_events)


def test_multi_event_step_is_rejected_without_mutation() -> None:
    """Do not collapse multiple source passages into one binary sample."""
    neuron = ThetaNeuron(theta=0.25, dt=1.0)
    with pytest.raises(ValueError, match="more than one source event"):
        neuron.simulate_complete(1, 16.0, "python")
    assert neuron.theta == 0.25


@pytest.mark.parametrize("n_steps", [-1, 1.0, True])
def test_invalid_step_count_fails_before_mutation(n_steps: object) -> None:
    """Reject negative and non-integer step counts at the public boundary."""
    neuron = ThetaNeuron(theta=0.25)
    before = neuron.theta
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 0.0)
    assert neuron.theta == before


def test_invalid_backend_fails_before_mutation() -> None:
    """Reject unknown dispatch selectors instead of silently using Python."""
    neuron = ThetaNeuron(theta=0.25)
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 0.0, backend="cuda")
    assert neuron.theta == 0.25


def test_non_finite_current_fails_before_mutation() -> None:
    """Apply the finite-input boundary to every dispatcher path."""
    neuron = ThetaNeuron(theta=0.25)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan, backend="auto")
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.nan)
    assert neuron.theta == 0.25


def test_exact_flow_rejects_overflow_without_mutation() -> None:
    """Translate a non-finite analytic candidate into mutation-free failure."""
    neuron = ThetaNeuron(theta=0.25, dt=1.0e308)
    before = neuron.theta
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(-1.0e308)
    assert neuron.theta == before


def test_reset_restores_only_the_runtime_state() -> None:
    """Restore phase while retaining the configured integration step."""
    neuron = _configured()
    expected_dt = neuron.dt
    neuron.step(2.2)
    neuron.reset()
    assert neuron.theta == 0.0
    assert neuron.dt == expected_dt
