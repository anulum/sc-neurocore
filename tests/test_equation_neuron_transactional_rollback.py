# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Transactional rollback for EquationNeuron step failures

"""Verify atomic rollback of EquationNeuron state, spike value and RNG streams."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS, EquationNeuron
from sc_neurocore.neurons.equation_refusals import (
    EquationDivisionFailure,
    EquationStateFailure,
)


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_deterministic_integration_failure_restores_committed_state_and_counters(
    method: str,
) -> None:
    """Restore state, graded spike value and step counter when integration diverges."""
    rhs = "v + I" if method == "map" else "-0.5 * v + I"
    neuron = EquationNeuron(
        equations={"v": rhs},
        state={"v": 0.4},
        dt=1.0,
        method=method,
        threshold="v > 0.5",
        reset={"v": "0.25"},
        spike_value="2.0 * v",
    )
    first_spike = neuron.step(I=0.5)
    assert first_spike == 1
    committed_state = dict(neuron.state)
    committed_graded = neuron.last_spike_value
    committed_random_step = neuron._random.step
    assert committed_graded > 0.0

    with pytest.raises(EquationStateFailure):
        neuron.step(I=1e308 * 2.0)

    assert neuron.state == committed_state
    assert neuron.last_spike_value == committed_graded
    assert neuron._random.step == committed_random_step

    twin = EquationNeuron(
        equations={"v": rhs},
        state={"v": 0.4},
        dt=1.0,
        method=method,
        threshold="v > 0.5",
        reset={"v": "0.25"},
        spike_value="2.0 * v",
    )
    twin.step(I=0.5)
    assert neuron.step(I=0.1) == twin.step(I=0.1)
    assert neuron.state == twin.state
    assert neuron.last_spike_value == twin.last_spike_value


def test_multistep_and_gauss_seidel_partial_updates_roll_back_atomically() -> None:
    """Discard earlier substep and earlier Gauss-Seidel variable mutations on fault."""
    substepped = EquationNeuron(
        equations={"v": "1.0 / (v - 2.0) + I"},
        state={"v": 0.0},
        dt=1.0,
        method="map",
        substeps=4,
    )
    substepped.step(I=0.5)
    before_substeps = dict(substepped.state)
    xi_substep_before = substepped._xi_substep

    with pytest.raises(EquationDivisionFailure, match="divides by zero"):
        substepped.step(I=2.5)

    assert substepped.state == before_substeps
    assert substepped._xi_substep == xi_substep_before

    sequential = EquationNeuron(
        equations={"u": "1.0", "v": "1.0 / (u - 1.1)"},
        state={"u": 1.0, "v": 0.25},
        dt=0.1,
        method="gauss_seidel",
    )
    with pytest.raises(EquationDivisionFailure, match="divides by zero"):
        sequential.step()

    assert sequential.state == {"u": 1.0, "v": 0.25}


def test_deterministic_threshold_fault_restores_state_and_graded_spike() -> None:
    """Roll back integrated state and last_spike_value when threshold evaluation faults."""
    neuron = EquationNeuron(
        equations={"v": "I"},
        state={"v": 2.0},
        dt=1.0,
        method="map",
        threshold="1.0 / (v - 1.0) > 0.0",
        reset={"v": "0.0"},
        spike_value="3.0 * v",
    )
    assert neuron.step(I=2.0) == 1
    assert neuron.last_spike_value == 6.0
    assert neuron.state == {"v": 0.0}

    with pytest.raises(EquationDivisionFailure, match="divides by zero"):
        neuron.step(I=1.0)

    assert neuron.state == {"v": 0.0}
    assert neuron.last_spike_value == 6.0


def test_edge_crossing_flag_and_partial_reset_roll_back_on_post_spike_fault() -> None:
    """Preserve rising-edge history and multi-variable state when spike handling faults."""
    oscillator = EquationNeuron(
        equations={"v": "I"},
        state={"v": 0.0},
        dt=1.0,
        method="map",
        threshold="v > 0.5",
        detection="crossing",
        spike_value="v / (I - 1.0)",
    )
    with pytest.raises(EquationDivisionFailure, match="divides by zero"):
        oscillator.step(I=1.0)

    assert oscillator.state == {"v": 0.0}
    assert oscillator._prev_threshold_active is False
    assert oscillator.step(I=2.0) == 1
    assert oscillator.last_spike_value == 2.0

    multi_reset = EquationNeuron(
        equations={"v": "I", "w": "w + 0.5"},
        state={"v": 0.0, "w": 1.0},
        dt=1.0,
        method="map",
        threshold="v > 0.5",
        reset={"v": "-1.0", "w": "1.0 / (I - 1.0)"},
        spike_value="v + w",
    )
    with pytest.raises(EquationDivisionFailure, match="divides by zero"):
        multi_reset.step(I=1.0)

    assert multi_reset.state == {"v": 0.0, "w": 1.0}
    assert multi_reset.last_spike_value == 0.0


@pytest.mark.parametrize("threshold_rng", ["lfsr16", "philox"])
def test_nonfinite_spike_value_and_reset_outputs_fail_closed_and_roll_back(
    threshold_rng: str,
) -> None:
    """Refuse non-finite graded spikes and reset states across threshold modes."""
    det_bad_grade = EquationNeuron(
        equations={"v": "I"},
        parameters={"scale": 1e308},
        state={"v": 0.25},
        dt=1.0,
        method="map",
        threshold="v > 0.5",
        reset={"v": "0.0"},
        spike_value="scale * 20.0 * v",
    )
    with pytest.raises(EquationStateFailure, match="spike_value must remain finite"):
        det_bad_grade.step(I=1.0)
    assert det_bad_grade.state == {"v": 0.25}
    assert det_bad_grade.last_spike_value == 0.0

    stoch_bad_reset = EquationNeuron(
        equations={"v": "v + 0.1"},
        parameters={"scale": 1e308},
        state={"v": 0.5},
        dt=0.1,
        detection="poisson",
        probability_expression="1.0",
        threshold_rng=threshold_rng,
        reset={"v": "scale * 20.0"},
        spike_value="2.0",
    )
    rng_before = stoch_bad_reset.stochastic_rng_state
    with pytest.raises(EquationStateFailure, match="became non-finite"):
        stoch_bad_reset.step()
    assert stoch_bad_reset.state == {"v": 0.5}
    assert stoch_bad_reset.last_spike_value == 0.0
    assert stoch_bad_reset.stochastic_rng_state == rng_before


def test_caller_noise_rng_stream_is_restored_when_step_fails() -> None:
    """Restore caller-supplied numpy Generator state after a failed noisy step."""
    neuron = EquationNeuron(
        equations={"v": "0.1 * xi + I / (I - 2.0)"},
        state={"v": 0.0},
        dt=0.25,
        substeps=2,
        noise_rng=np.random.default_rng(20261124),
    )
    neuron.step(I=1.0)
    committed = dict(neuron.state)

    with pytest.raises(EquationDivisionFailure, match="divides by zero"):
        neuron.step(I=2.0)

    assert neuron.state == committed

    reference = EquationNeuron(
        equations={"v": "0.1 * xi + I / (I - 2.0)"},
        state={"v": 0.0},
        dt=0.25,
        substeps=2,
        noise_rng=np.random.default_rng(20261124),
    )
    reference.step(I=1.0)
    neuron.step(I=0.5)
    reference.step(I=0.5)
    assert neuron.state == reference.state
