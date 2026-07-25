# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing inverse error calculators

from __future__ import annotations

from sc_neurocore.core import (
    bernoulli_std_error,
    bipolar_std_error,
    hoeffding_confidence,
    hoeffding_min_length,
    min_length_for_std_error,
)


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
