# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing error-bound input validation

from __future__ import annotations

import pytest

from sc_neurocore.core import (
    bernoulli_variance,
    bipolar_variance,
    dot_product_variance,
    hoeffding_confidence,
    hoeffding_min_length,
    min_length_for_std_error,
    multiply_correlation_bias,
)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_probability_domain_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        bernoulli_variance(bad, 10)


@pytest.mark.parametrize("bad", [-1.1, 1.1, float("nan")])
def test_bipolar_domain_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        bipolar_variance(bad, 10)


@pytest.mark.parametrize("bad_len", [0, -5])
def test_nonpositive_length_rejected(bad_len: int) -> None:
    with pytest.raises(ValueError):
        bernoulli_variance(0.5, bad_len)


def test_boolean_length_rejected() -> None:
    with pytest.raises(TypeError):
        bernoulli_variance(0.5, True)  # bool is not an accepted int length


def test_scc_domain_rejected() -> None:
    with pytest.raises(ValueError):
        multiply_correlation_bias(0.5, 0.5, 1.5)


def test_dot_product_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        dot_product_variance([0.5, 0.5], [0.5], 10)


def test_dot_product_empty_rejected() -> None:
    with pytest.raises(ValueError):
        dot_product_variance([], [], 10)


@pytest.mark.parametrize("bad_eps", [0.0, -0.1, 1.5, float("nan")])
def test_hoeffding_epsilon_domain_rejected(bad_eps: float) -> None:
    with pytest.raises(ValueError):
        hoeffding_min_length(bad_eps, 0.9)


@pytest.mark.parametrize("bad_conf", [-0.1, 1.0, 1.5])
def test_hoeffding_confidence_domain_rejected(bad_conf: float) -> None:
    with pytest.raises(ValueError):
        hoeffding_min_length(0.05, bad_conf)


def test_min_length_nonpositive_std_rejected() -> None:
    with pytest.raises(ValueError):
        min_length_for_std_error(0.5, 0.0)


def test_hoeffding_confidence_bad_epsilon_rejected() -> None:
    with pytest.raises(ValueError):
        hoeffding_confidence(100, 0.0)
