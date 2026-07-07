# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Analytic SC error-bound contract tests

"""Contract and Monte-Carlo tests for :mod:`sc_neurocore.core.sc_error_bounds`.

The analytic bounds are validated both against their closed forms and against
Monte-Carlo simulation of the real production encoders (``rate_encode`` and
``core.bipolar``), so the formulas are tied to actual bitstream behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.core import (
    SCErrorBound,
    bernoulli_std_error,
    bernoulli_variance,
    bipolar_std_error,
    bipolar_variance,
    dot_product_variance,
    hoeffding_confidence,
    hoeffding_min_length,
    low_discrepancy_error_bound,
    min_length_for_std_error,
    multiply_correlation_bias,
    multiply_variance,
    mux_add_variance,
    sc_error_bound,
)
from sc_neurocore.core.bipolar import bipolar_decode, bipolar_encode
from sc_neurocore.encoding.encoders import rate_encode


# --- closed-form correctness -------------------------------------------------


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


# --- correlation bias (SCC) --------------------------------------------------


def test_correlation_bias_zero_when_independent() -> None:
    assert multiply_correlation_bias(0.6, 0.4, 0.0) == pytest.approx(0.0)


def test_correlation_bias_positive_reaches_comonotone_bound() -> None:
    pa, pb = 0.6, 0.4
    # rho = +1 -> E[AND] = min(pa, pb); bias = min - pa*pb.
    assert multiply_correlation_bias(pa, pb, 1.0) == pytest.approx(min(pa, pb) - pa * pb)


def test_correlation_bias_negative_reaches_countermonotone_bound() -> None:
    pa, pb = 0.6, 0.7
    # rho = -1 -> E[AND] = max(0, pa+pb-1); bias = that - pa*pb (negative).
    expected = max(0.0, pa + pb - 1.0) - pa * pb
    assert multiply_correlation_bias(pa, pb, -1.0) == pytest.approx(expected)


# --- inverse calculators -----------------------------------------------------


def test_hoeffding_min_length_meets_confidence() -> None:
    eps, conf = 0.05, 0.95
    n = hoeffding_min_length(eps, conf)
    assert isinstance(n, int)
    assert hoeffding_confidence(n, eps) >= conf
    # One shorter must not already satisfy the target (tightness).
    assert hoeffding_confidence(n - 1, eps) < conf


def test_hoeffding_confidence_monotone_in_length() -> None:
    eps = 0.1
    assert hoeffding_confidence(500, eps) > hoeffding_confidence(50, eps)


def test_min_length_for_std_error_unipolar_and_bipolar() -> None:
    n_uni = min_length_for_std_error(0.5, 0.01)
    assert bernoulli_std_error(0.5, n_uni) <= 0.01
    n_bip = min_length_for_std_error(0.0, 0.01, bipolar=True)
    assert bipolar_std_error(0.0, n_bip) <= 0.01


def test_min_length_for_std_error_floor_is_one() -> None:
    # value at the extreme has zero variance -> length floored at 1.
    assert min_length_for_std_error(0.0, 0.01) == 1


# --- SCErrorBound summary ----------------------------------------------------


def test_sc_error_bound_summary_fields() -> None:
    bound = sc_error_bound(0.5, 100)
    assert isinstance(bound, SCErrorBound)
    assert bound.variance == pytest.approx(0.0025)
    assert bound.std_error == pytest.approx(0.05)
    assert bound.ci95_halfwidth == pytest.approx(1.959963984540054 * 0.05)
    assert bound.bipolar is False


def test_sc_error_bound_bipolar_uses_bipolar_variance() -> None:
    bound = sc_error_bound(0.0, 100, bipolar=True)
    assert bound.bipolar is True
    assert bound.variance == pytest.approx(1.0 / 100)


# --- validation (fail-closed) ------------------------------------------------


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


# --- Monte-Carlo validation against the real encoders ------------------------


def _empirical_unipolar_variance(p: float, n: int, trials: int, seed: int) -> float:
    """Empirical Var(k/N) using the production ``rate_encode`` Bernoulli path."""
    streams = rate_encode(np.full(trials, p, dtype=np.float64), T=n, seed=seed)
    estimates = streams.mean(axis=0)  # one decoded value per independent column
    return float(np.var(estimates))


def test_monte_carlo_matches_bernoulli_variance() -> None:
    p, n, trials = 0.3, 400, 20000
    empirical = _empirical_unipolar_variance(p, n, trials, seed=7)
    analytic = bernoulli_variance(p, n)
    assert empirical == pytest.approx(analytic, rel=0.15)


def test_monte_carlo_matches_bipolar_variance() -> None:
    v, n, trials = 0.4, 400, 6000
    rng = np.random.default_rng(11)
    estimates = np.array(
        [bipolar_decode(bipolar_encode(v, n, rng)) for _ in range(trials)],
        dtype=np.float64,
    )
    empirical = float(np.var(estimates))
    assert empirical == pytest.approx(bipolar_variance(v, n), rel=0.15)


def test_monte_carlo_matches_multiply_variance_independent() -> None:
    pa, pb, n, trials = 0.6, 0.5, 400, 20000
    a = rate_encode(np.full(trials, pa, dtype=np.float64), T=n, seed=1)
    b = rate_encode(np.full(trials, pb, dtype=np.float64), T=n, seed=2)
    anded = (a.astype(np.int32) & b.astype(np.int32)).mean(axis=0)
    empirical = float(np.var(anded))
    assert empirical == pytest.approx(multiply_variance(pa, pb, n), rel=0.2)


def test_monte_carlo_correlated_and_hits_min_bound() -> None:
    # Fully correlated (shared random source) -> E[AND] = min(pa, pb).
    pa, pb, n, trials = 0.6, 0.4, 400, 20000
    rng = np.random.default_rng(5)
    u = rng.random((n, trials))
    a = (u < pa).astype(np.int32)
    b = (u < pb).astype(np.int32)
    anded_mean = float((a & b).mean())
    independent = pa * pb
    predicted = independent + multiply_correlation_bias(pa, pb, 1.0)
    assert predicted == pytest.approx(min(pa, pb))
    assert anded_mean == pytest.approx(min(pa, pb), abs=0.01)
