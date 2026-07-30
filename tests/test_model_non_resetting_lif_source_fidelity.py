# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Source-equation tests for the Kobayashi MAT(1) specialization."""

from __future__ import annotations

import hashlib
import math
import struct

import pytest

from sc_neurocore.neurons.models.non_resetting_lif import NonResettingLIFNeuron


def _source_inputs() -> list[float]:
    """Return the deterministic enrolled MAT(1) current sequence."""
    return [0.0] * 32 + [0.7] * 8192 + [value for _ in range(1024) for value in (0.2, 0.9)]


def _independent_trace() -> tuple[str, list[int], tuple[float, float, float]]:
    """Evaluate a direct scalar transcription without production imports."""
    v = theta = refractory = 0.0
    digest = hashlib.sha256()
    events: list[int] = []
    for index, current in enumerate(_source_inputs()):
        v += 0.001 * (-v + 50.0 * current) / 5.0
        theta *= math.exp(-0.001 / 50.0)
        refractory = max(0.0, refractory - 0.001)
        spike = refractory == 0.0 and v >= 19.0 + theta
        if spike:
            theta += 37.0
            refractory = 2.0
            events.append(index)
        digest.update(struct.pack("<dddB", v, theta, refractory, int(spike)))
    return digest.hexdigest(), events, (v, theta, refractory)


def test_mat1_defaults_and_source_timescale() -> None:
    """Expose the paper's MAT(1) time scale and explicit specialization."""
    neuron = NonResettingLIFNeuron()
    assert (neuron.v, neuron.theta, neuron.refractory_remaining) == (0.0, 0.0, 0.0)
    assert neuron.tau_theta == 50.0
    assert neuron.refractory_period == 2.0
    assert neuron.dt == 0.001
    assert neuron.threshold == 19.0


def test_one_step_matches_source_equations() -> None:
    """Match equation 1 and the single equation 2-3 history exactly."""
    neuron = NonResettingLIFNeuron(v=2.0, theta=3.0)
    expected_v = 2.0 + neuron.dt * (-2.0 + neuron.resistance * 0.5) / neuron.tau_m
    expected_theta = 3.0 * math.exp(-neuron.dt / neuron.tau_theta)
    assert neuron.step(0.5) == 0
    assert neuron.v == expected_v
    assert neuron.theta == expected_theta


def test_spike_does_not_reset_voltage_and_starts_refractory() -> None:
    """Preserve voltage while raising history and starting the 2 ms gate."""
    neuron = NonResettingLIFNeuron(v=20.0)
    expected_v = 20.0 + neuron.dt * (-20.0) / neuron.tau_m
    assert neuron.step(0.0) == 1
    assert neuron.v == expected_v
    assert neuron.theta == 37.0
    assert neuron.refractory_remaining == 2.0
    assert neuron.step(0.0) == 0


def test_enrolled_trace_matches_independent_receipt() -> None:
    """Match the independent complete-state source receipt."""
    expected_digest, expected_events, expected_final = _independent_trace()
    neuron = NonResettingLIFNeuron()
    digest = hashlib.sha256()
    events: list[int] = []
    for index, current in enumerate(_source_inputs()):
        spike = neuron.step(current)
        if spike:
            events.append(index)
        digest.update(
            struct.pack(
                "<dddB",
                neuron.v,
                neuron.theta,
                neuron.refractory_remaining,
                spike,
            )
        )
    assert expected_digest == "2ac13e42322a3ac6b4059f29190f0936409c9d4bf28f1837e4bee97add2069c6"
    assert digest.hexdigest() == expected_digest
    assert events == expected_events == [3945]
    assert (neuron.v, neuron.theta, neuron.refractory_remaining) == expected_final


@pytest.mark.parametrize(
    "field",
    ["v", "theta", "omega", "tau_m", "tau_theta", "alpha", "resistance", "refractory_period", "dt"],
)
def test_invalid_state_fails_before_mutation(field: str) -> None:
    """Reject corrupted runtime state atomically."""
    neuron = NonResettingLIFNeuron()
    before = (neuron.v, neuron.theta, neuron.refractory_remaining)
    setattr(neuron, field, math.nan)
    with pytest.raises(ValueError):
        neuron.step(0.7)
    if field not in {"v", "theta"}:
        assert (neuron.v, neuron.theta, neuron.refractory_remaining) == before


def test_nonfinite_input_fails_atomically() -> None:
    """Reject a non-finite drive without partial state mutation."""
    neuron = NonResettingLIFNeuron(v=1.0, theta=2.0)
    before = (neuron.v, neuron.theta, neuron.refractory_remaining)
    with pytest.raises(ValueError, match="input current"):
        neuron.step(math.inf)
    assert (neuron.v, neuron.theta, neuron.refractory_remaining) == before
