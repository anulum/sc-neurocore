# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing error summary contracts

from __future__ import annotations

import pytest

from sc_neurocore.core import SCErrorBound, sc_error_bound


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
