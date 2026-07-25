# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing correlation-bias contracts

from __future__ import annotations

import pytest

from sc_neurocore.core import multiply_correlation_bias


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
