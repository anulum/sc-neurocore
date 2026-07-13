# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Published DPI Python reference contract tests

"""Fidelity and safety tests for the coupled current-mode DPI recurrence."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.dpi_neuron import DPINeuron


def _configured() -> DPINeuron:
    """Return a stable non-default contract exercising every maintained field."""
    return DPINeuron(
        i_mem=0.37,
        i_ahp=0.08,
        refractory_time=0.0,
        i_threshold=1.3,
        i_reset=0.2,
        i_rest=0.15,
        i_tau=0.9,
        i_g=1.4,
        i_tau_ahp=0.12,
        i_ga=0.8,
        i_spike=4.2,
        i_0=0.02,
        kappa=0.65,
        alpha=8.0,
        tau=7.0,
        tau_ahp=45.0,
        refractory_period=0.6,
        dt=0.05,
    )


def _events(neuron: DPINeuron, current: float, steps: int) -> list[int]:
    """Return zero-based spike indices from one direct reference-model run."""
    return [index for index in range(steps) if neuron.step(current) == 1]


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"i_mem": 0.0},
        {"i_mem": math.nan},
        {"i_ahp": -1.0},
        {"refractory_time": -0.1},
        {"i_threshold": math.inf},
        {"i_threshold": 0.0},
        {"i_reset": 1.0},
        {"i_rest": -0.1},
        {"i_tau": 0.0},
        {"i_g": 0.0},
        {"i_tau_ahp": 0.0},
        {"i_ga": 0.0},
        {"i_spike": 0.0},
        {"i_0": 0.0},
        {"kappa": 0.0},
        {"alpha": 0.0},
        {"tau": 0.0},
        {"tau_ahp": 0.0},
        {"refractory_period": 0.05},
        {"dt": 0.0},
    ],
)
def test_constructor_rejects_nonphysical_contract(overrides: dict[str, float]) -> None:
    """Reject every invalid maintained state/parameter family at construction."""
    with pytest.raises(ValueError):
        DPINeuron(**cast(Any, overrides))


@pytest.mark.parametrize(
    ("field", "value"),
    [("i_mem", math.nan), ("i_ahp", -1.0), ("i_ga", 0.0), ("dt", 0.0)],
)
def test_mutated_runtime_contract_is_revalidated(field: str, value: float) -> None:
    """Do not assume dataclass construction is the only mutation boundary."""
    neuron = DPINeuron()
    setattr(neuron, field, value)
    with pytest.raises(ValueError):
        neuron.step(0.0)


@pytest.mark.parametrize("current", [math.nan, math.inf, math.ulp(math.inf)])
def test_non_finite_input_fails_without_mutation(current: float) -> None:
    """Reject non-finite input before evaluating either coupled equation."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_negative_total_input_fails_without_mutation() -> None:
    """Keep the source current inside the physical non-negative domain."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match=r"i_rest \+ current"):
        neuron.step(-0.1000001)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_non_finite_membrane_candidate_fails_without_reset_masking() -> None:
    """Validate the Euler candidate before a threshold reset can hide overflow."""
    neuron = DPINeuron(tau=float.fromhex("0x1.0p-1022"))
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(float.fromhex("0x1.fffffffffffffp+1023"))
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_nonlinear_evaluation_failure_is_translated_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contain a nonlinear-domain failure behind the public value contract."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)

    def fail_derivatives(
        _self: DPINeuron,
        _current: float,
        *,
        spike_active: bool,
    ) -> tuple[float, float]:
        assert not spike_active
        raise OverflowError("nonlinear circuit overflow")

    monkeypatch.setattr(DPINeuron, "_derivatives", fail_derivatives)
    with pytest.raises(ValueError, match="nonlinear current evaluation failed"):
        neuron.step(0.0)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_non_finite_adaptation_candidate_fails_without_mutation() -> None:
    """Reject arithmetic overflow across the simultaneous Euler candidate set."""
    neuron = DPINeuron(
        i_ahp=float.fromhex("0x1.fffffffffffffp+1023"),
        refractory_time=2.0,
        i_tau_ahp=1.0,
        tau_ahp=1.0,
        refractory_period=2.0,
        dt=2.0,
    )
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="Euler update must remain finite"):
        neuron.step(0.0)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_nonphysical_adaptation_candidate_fails_without_mutation() -> None:
    """Reject a negative post-Euler AHP current atomically."""
    neuron = DPINeuron(i_ahp=0.01, tau_ahp=0.01, dt=0.1)
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="physical current domain"):
        neuron.step(0.0)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


@pytest.mark.parametrize("n_steps", [-1, 1.0, True])
def test_invalid_step_count_fails_before_mutation(n_steps: object) -> None:
    """Require a non-negative integer at the public simulation boundary."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 0.0)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_invalid_backend_and_total_current_fail_before_mutation() -> None:
    """Reject dispatch and current-domain errors without fallback mutation."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 0.0, backend="cuda")
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan)
    with pytest.raises(ValueError, match="finite and non-negative"):
        neuron.simulate(1, -0.2)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_rust_compatibility_boundary_is_exact_factory_contract() -> None:
    """Use the fixed-constructor PyO3 engine only for an exact field match."""
    neuron = DPINeuron()
    assert neuron._matches_rust_engine_contract()
    neuron.i_ahp = math.nextafter(neuron.i_ahp, math.inf)
    assert not neuron._matches_rust_engine_contract()


def test_reset_restores_current_baselines_and_preserves_parameters() -> None:
    """Reset three dynamic states without destroying circuit configuration."""
    neuron = _configured()
    parameters = tuple(
        neuron.__dict__[name]
        for name in neuron.__dict__
        if name
        not in {
            "i_mem",
            "i_ahp",
            "refractory_time",
        }
    )
    neuron.simulate(100, 5.0, backend="python")
    neuron.reset()
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == (
        neuron.i_reset,
        neuron.i_0,
        0.0,
    )
    assert (
        tuple(
            neuron.__dict__[name]
            for name in neuron.__dict__
            if name
            not in {
                "i_mem",
                "i_ahp",
                "refractory_time",
            }
        )
        == parameters
    )


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
