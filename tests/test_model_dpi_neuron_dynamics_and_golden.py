# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (dynamics_and_golden) from former test_model_dpi_neuron.py

from __future__ import annotations

from tests.model_dpi_neuron_support import *  # noqa: F403

def test_defaults_and_one_step_match_published_coupled_equations() -> None:
    """Anchor Eq. (2) feedback and both Eq. (3) Euler increments."""
    neuron = DPINeuron()
    assert neuron == DPINeuron(
        0.01,
        0.01,
        0.0,
        1.0,
        0.01,
        0.1,
        1.0,
        1.0,
        0.1,
        1.0,
        5.0,
        0.01,
        0.7,
        10.0,
        20.0,
        100.0,
        2.0,
        0.1,
    )
    expected_feedback = 5.017216468376423e-07
    assert neuron._feedback_current(neuron.i_mem) == pytest.approx(expected_feedback, abs=1.0e-21)
    assert neuron.step(5.0) == 0
    assert neuron.i_mem == pytest.approx(0.010201975272610835, abs=1.0e-17)
    assert neuron.i_ahp == pytest.approx(0.00999, abs=1.0e-17)
    assert neuron.refractory_time == 0.0


def test_feedback_gate_uses_stable_negative_and_positive_branches() -> None:
    """Exercise both stable logistic branches without altering the equation."""
    low = DPINeuron(i_mem=0.01, i_threshold=50.0)
    high = DPINeuron(i_mem=100.0, i_threshold=1.0)
    assert math.isfinite(low._feedback_current(low.i_mem))
    assert low._feedback_current(low.i_mem) > 0.0
    assert math.isfinite(high._feedback_current(high.i_mem))
    assert high._feedback_current(high.i_mem) > low._feedback_current(low.i_mem)


def test_threshold_crossing_resets_and_starts_pulse_after_euler_step() -> None:
    """Apply the post-update threshold/reset ordering used by schema and RTL."""
    neuron = DPINeuron(i_mem=0.99)
    assert neuron.step(10.0) == 1
    assert neuron.i_mem == neuron.i_reset
    assert neuron.refractory_time == neuron.refractory_period


def test_refractory_pulse_holds_membrane_and_drives_adaptation() -> None:
    """Use r(t)=1 during the refractory pulse in the adaptation DPI."""
    neuron = DPINeuron(refractory_time=2.0)
    before_ahp = neuron.i_ahp
    assert neuron.step(0.0) == 0
    assert neuron.i_mem == neuron.i_reset
    assert neuron.i_ahp > before_ahp
    assert neuron.refractory_time == pytest.approx(1.9)


def test_short_final_refractory_step_lands_exactly_at_zero() -> None:
    """Do not leave a negative or sub-step timer residue."""
    neuron = DPINeuron(refractory_time=0.05)
    neuron.step(0.0)
    assert neuron.refractory_time == 0.0


def test_adaptation_decays_between_spike_pulses() -> None:
    """With r(t)=0, the adaptation DPI relaxes toward zero current."""
    neuron = DPINeuron()
    neuron.step(0.0)
    assert neuron.i_ahp == pytest.approx(0.00999, abs=1.0e-17)


def test_configured_python_trace_matches_enrolled_golden() -> None:
    """Bind the complete configurable contract to one stable trace endpoint."""
    neuron = _configured()
    trace, spikes = neuron.simulate(400, 5.0, backend="python")
    assert trace.shape == (400,)
    assert trace.dtype == np.float64
    assert spikes == 4
    assert neuron.i_mem == trace[-1] == 0.2
    assert neuron.i_ahp == pytest.approx(0.27412306389119817, abs=2.0e-15)
    assert neuron.refractory_time == 0.0


def test_empty_python_trace_is_identity_for_all_states() -> None:
    """Treat a zero-step run as a genuine identity operation."""
    neuron = _configured()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    trace, spikes = neuron.simulate(0, 5.0, backend="python")
    assert trace.shape == (0,)
    assert trace.dtype == np.float64
    assert spikes == 0
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_sustained_drive_exhibits_spike_frequency_adaptation() -> None:
    """Prove the spike-triggered AHP lengthens inter-spike intervals."""
    events = _events(DPINeuron(), current=5.0, steps=5_000)
    intervals = np.diff(events)
    assert events[:3] == [295, 612, 931]
    assert intervals[-1] > intervals[0]


def test_drive_increases_spike_count_without_saturation() -> None:
    """Exercise the nonlinear FI response at three physical operating points."""
    counts = [len(_events(DPINeuron(), current=value, steps=5_000)) for value in (3.0, 5.0, 10.0)]
    assert 0 < counts[0] < counts[1] < counts[2] < 5_000


def test_long_run_remains_finite_and_physical() -> None:
    """Keep both circuit currents and the pulse timer in their physical domain."""
    neuron = DPINeuron()
    for _ in range(20_000):
        neuron.step(5.0)
    assert math.isfinite(neuron.i_mem) and neuron.i_mem > 0.0
    assert math.isfinite(neuron.i_ahp) and neuron.i_ahp >= 0.0
    assert math.isfinite(neuron.refractory_time) and neuron.refractory_time >= 0.0
