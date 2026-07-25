# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-faithful IQIF dynamics contracts

"""Source calibration and piecewise dynamics tests for IQIF."""

from __future__ import annotations

import dataclasses

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron

from .model_iqif_support import _trace


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
