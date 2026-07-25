# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing Monte Carlo error validation

"""Compare analytic error bounds with real production bitstream encoders."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core import (
    bernoulli_variance,
    bipolar_variance,
    multiply_correlation_bias,
    multiply_variance,
)
from sc_neurocore.core.bipolar import bipolar_decode, bipolar_encode
from sc_neurocore.encoding.encoders import rate_encode
from tests.sc_error_bounds_support import empirical_unipolar_variance


def test_monte_carlo_matches_bernoulli_variance() -> None:
    p, n, trials = 0.3, 400, 20000
    empirical = empirical_unipolar_variance(p, n, trials, seed=7)
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
