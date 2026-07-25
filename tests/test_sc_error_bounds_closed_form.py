# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing closed-form error contracts

from __future__ import annotations

import math

import pytest

from sc_neurocore.core import (
    bernoulli_std_error,
    bernoulli_variance,
    bipolar_std_error,
    bipolar_variance,
    dot_product_variance,
    low_discrepancy_error_bound,
    multiply_variance,
    mux_add_variance,
)


def test_bernoulli_variance_closed_form() -> None:
    assert bernoulli_variance(0.5, 100) == pytest.approx(0.25 / 100)
    assert bernoulli_variance(0.2, 50) == pytest.approx(0.2 * 0.8 / 50)
    assert bernoulli_std_error(0.5, 100) == pytest.approx(math.sqrt(0.0025))


def test_bernoulli_variance_zero_at_extremes() -> None:
    assert bernoulli_variance(0.0, 10) == 0.0
    assert bernoulli_variance(1.0, 10) == 0.0


def test_bipolar_variance_closed_form() -> None:
    assert bipolar_variance(0.0, 100) == pytest.approx(1.0 / 100)
    assert bipolar_variance(0.5, 40) == pytest.approx((1 - 0.25) / 40)
    assert bipolar_variance(1.0, 10) == 0.0
    assert bipolar_variance(-1.0, 10) == 0.0
    assert bipolar_std_error(0.0, 100) == pytest.approx(0.1)


def test_multiply_variance_matches_bernoulli_of_product() -> None:
    pa, pb, n = 0.6, 0.4, 200
    product = pa * pb
    assert multiply_variance(pa, pb, n) == pytest.approx(product * (1 - product) / n)


def test_mux_add_variance_uses_mean_value() -> None:
    pa, pb, n = 0.3, 0.7, 128
    q = 0.5
    assert mux_add_variance(pa, pb, n) == pytest.approx(q * (1 - q) / n)


def test_low_discrepancy_bound_is_one_over_n() -> None:
    assert low_discrepancy_error_bound(256) == pytest.approx(1.0 / 256)


def test_dot_product_variance_closed_form() -> None:
    a = [0.5, 0.5]
    b = [0.5, 0.5]
    n = 100
    expected = (0.25 * 0.75 + 0.25 * 0.75) / (4 * n)
    assert dot_product_variance(a, b, n) == pytest.approx(expected)
