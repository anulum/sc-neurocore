# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import math

import pytest

from sc_neurocore.utils.numerics import (
    boltzmann,
    boltzmann_inv,
    clip_gating,
    clip_voltage,
    safe_cosh,
    safe_exp,
    safe_tanh,
)


def test_safe_exponential_and_hyperbolic_functions_remain_finite_at_extreme_voltage():
    assert math.isfinite(safe_exp(1_000.0))
    assert safe_exp(1_000.0) == pytest.approx(math.exp(500.0))
    assert safe_exp(-1_000.0) == pytest.approx(math.exp(-500.0))
    assert math.isfinite(safe_cosh(1_000.0))
    assert safe_tanh(1_000.0) == pytest.approx(1.0)
    assert safe_tanh(-1_000.0) == pytest.approx(-1.0)


def test_boltzmann_curves_are_bounded_and_monotone_around_half_activation():
    low = boltzmann(-80.0, v_half=-40.0, k=6.0)
    mid = boltzmann(-40.0, v_half=-40.0, k=6.0)
    high = boltzmann(20.0, v_half=-40.0, k=6.0)

    assert 0.0 < low < mid < high < 1.0
    assert mid == pytest.approx(0.5)
    assert boltzmann_inv(-40.0, v_half=-40.0, k=6.0) == pytest.approx(0.5)


def test_gating_and_voltage_clamps_enforce_physiological_bounds():
    assert clip_gating(-0.25) == 0.0
    assert clip_gating(0.5) == 0.5
    assert clip_gating(1.25) == 1.0
    assert clip_voltage(-500.0) == -200.0
    assert clip_voltage(50.0) == 50.0
    assert clip_voltage(250.0, v_min=-80.0, v_max=40.0) == 40.0
