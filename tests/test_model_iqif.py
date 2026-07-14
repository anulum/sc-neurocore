# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-faithful Wu et al. 2021 IQIF model contracts

"""Exact state, validation, reset, batch, and dynamics tests for IQIF."""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron


def _trace(neuron: IntegerQIFNeuron, steps: int, current: int) -> tuple[list[int], list[int]]:
    values: list[int] = []
    spikes: list[int] = []
    for index in range(steps):
        if neuron.step(current):
            spikes.append(index)
        values.append(neuron.v)
    return values, spikes


def test_defaults_and_branch_point_match_pinned_source() -> None:
    """The public constructor is the 2021 repository tutorial contract."""
    neuron = IntegerQIFNeuron()
    assert dataclasses.is_dataclass(neuron)
    assert (
        neuron.v,
        neuron.v_rest,
        neuron.v_threshold,
        neuron.v_reset,
        neuron.a,
        neuron.b,
        neuron.v_max,
        neuron.v_min,
    ) == (128, 128, 200, 128, 1, 1, 255, 0)
    assert neuron.branch_point == 164
    assert neuron.dt == 1.0
    assert neuron.SLOPE_FRACTION_BITS == 3


def test_source_tutorial_trace_is_exact() -> None:
    """The 400-tick source tutorial has its exact 15-step orbit and features."""
    values, spike_indices = _trace(IntegerQIFNeuron(), 400, 10)
    assert values[:15] == [
        138,
        146,
        153,
        159,
        165,
        170,
        176,
        183,
        190,
        198,
        207,
        217,
        229,
        242,
        128,
    ]
    assert spike_indices == list(range(14, 400, 15))
    assert len(spike_indices) == 26
    assert (min(values), max(values), values[-1], sum(values)) == (128, 242, 198, 71_904)
    assert sum(values) / len(values) == 179.76


def test_piecewise_force_uses_pre_step_state_and_arithmetic_q03_shift() -> None:
    """Both restoring-force branches use the source's signed arithmetic shift."""
    lower = IntegerQIFNeuron(v=150)
    assert lower.branch_point == 164
    assert lower.step(0) == 0
    assert lower.v == 147  # 150 + ((128 - 150) >> 3)

    upper = IntegerQIFNeuron(v=201)
    assert upper.step(0) == 0
    assert upper.v == 201  # (201 - 200) >> 3 == 0

    upper.v = 208
    assert upper.step(0) == 0
    assert upper.v == 209  # (208 - 200) >> 3 == 1


def test_branch_point_uses_cpp_truncation_not_python_floor() -> None:
    """A negative non-integral numerator truncates toward zero like C++."""
    neuron = IntegerQIFNeuron(
        v=-20,
        v_rest=-20,
        v_threshold=11,
        v_reset=-20,
        a=2,
        b=1,
        v_max=100,
        v_min=-100,
    )
    assert (neuron.b * neuron.v_threshold + neuron.a * neuron.v_rest) == -29
    assert neuron.branch_point == -9
    assert -29 // 3 == -10


def test_spike_boundary_is_strict_and_reset_is_hard() -> None:
    """A candidate equal to v_max survives; v_max+1 emits and hard-resets."""
    equal = IntegerQIFNeuron(v=255)
    assert equal.step(-6) == 0
    assert equal.v == 255

    above = IntegerQIFNeuron(v=255)
    assert above.step(-5) == 1
    assert above.v == above.v_reset == 128


def test_lower_clamp_and_zero_coefficient_profiles_are_supported() -> None:
    """The lower bound is inclusive and source burst profiles may set one slope to zero."""
    neuron = IntegerQIFNeuron(v=0, a=0, b=3)
    assert neuron.step(-10) == 0
    assert neuron.v == neuron.v_min == 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"a": -1},
        {"b": -1},
        {"a": 0, "b": 0},
        {"v_min": 128},
        {"v_rest": 200},
        {"v_threshold": 255},
        {"v_reset": 256},
        {"v": -1},
        {"v_max": 1 << 31},
    ),
)
def test_constructor_rejects_invalid_contracts(kwargs: dict[str, int]) -> None:
    """Every ordering, coefficient, state, and width invariant fails closed."""
    with pytest.raises(ValueError):
        IntegerQIFNeuron(**kwargs)


@pytest.mark.parametrize("value", (True, 10.0, 10.5, "10", object()))
def test_current_requires_an_exact_integer_without_mutation(value: object) -> None:
    """No Boolean or Float64 contamination crosses the integer soma boundary."""
    neuron = IntegerQIFNeuron()
    before = neuron.v
    with pytest.raises(ValueError, match="current"):
        neuron.step(cast(int, value))
    assert neuron.v == before


def test_numpy_integer_is_accepted_and_normalised() -> None:
    """Integer scalar protocols remain usable without accepting float coercion."""
    neuron = IntegerQIFNeuron(v=np.int64(128))
    assert type(neuron.v) is int
    assert neuron.step(np.int32(10)) == 0
    assert neuron.v == 138


def test_runtime_parameter_corruption_fails_before_state_mutation() -> None:
    """Public dataclass mutation is revalidated on every state-changing call."""
    neuron = IntegerQIFNeuron()
    neuron.a = -1
    before = neuron.v
    with pytest.raises(ValueError, match="non-negative"):
        neuron.step(10)
    assert neuron.v == before


def test_reset_restores_rest_only() -> None:
    """Reset restores v_rest while preserving all configured parameters."""
    neuron = IntegerQIFNeuron(
        v=150,
        v_rest=100,
        v_threshold=180,
        v_reset=140,
        a=2,
        b=7,
        v_max=250,
        v_min=3,
    )
    parameters = (neuron.v_rest, neuron.v_threshold, neuron.v_reset, neuron.a, neuron.b)
    neuron.reset()
    assert neuron.v == 100
    assert (neuron.v_rest, neuron.v_threshold, neuron.v_reset, neuron.a, neuron.b) == parameters


def test_python_batch_is_exact_and_commits_final_state() -> None:
    """The public Python batch returns contiguous int64 state and exact events."""
    neuron = IntegerQIFNeuron()
    trace, spikes = neuron.simulate(400, 10, backend="python")
    assert trace.dtype == np.int64
    assert trace.flags.c_contiguous
    assert trace.shape == (400,)
    assert spikes == 26
    assert neuron.v == trace[-1] == 198


def test_empty_batch_preserves_state() -> None:
    """Zero ticks allocate no phantom state or event."""
    neuron = IntegerQIFNeuron(v=150)
    trace, spikes = neuron.simulate(0, 10, backend="python")
    assert trace.shape == (0,)
    assert spikes == 0
    assert neuron.v == 150


@pytest.mark.parametrize("n_steps", (-1, True, 1.0, 1 << 31))
def test_invalid_batch_length_fails_before_mutation(n_steps: object) -> None:
    """The public batch has one bounded integer length contract."""
    neuron = IntegerQIFNeuron()
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 10)
    assert neuron.v == 128


def test_invalid_backend_fails_before_mutation() -> None:
    """Unknown dispatch selectors cannot silently fall through to Python."""
    neuron = IntegerQIFNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 10, backend="cuda")
    assert neuron.v == 128


def test_firing_rate_is_monotonic_over_the_source_tutorial_currents() -> None:
    """The enrolled tonic regime increases event count with integer drive."""
    counts = []
    for current in (5, 10, 20, 40):
        _, spikes = IntegerQIFNeuron().simulate(3_000, current, backend="python")
        counts.append(spikes)
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)
