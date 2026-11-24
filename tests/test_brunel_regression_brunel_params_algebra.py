# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrunelParamsAlgebra from former test_brunel_regression.py

"""Focused suite: TestBrunelParamsAlgebra from former test_brunel_regression.py."""

from __future__ import annotations

import pytest
from tests.brunel_regression_support import (
    BrunelParams,
)


class TestBrunelParamsAlgebra:
    """Derived quantities of the Brunel parameters."""

    def test_n_total(self) -> None:
        """The population totals 1000 neurons."""
        bp = BrunelParams(n_exc=800, n_inh=200)
        assert bp.n_total == 1000

    def test_weight_inh(self) -> None:
        """The inhibitory weight is 2.0 for the defaults."""
        bp = BrunelParams(weight_exc=0.5, g_inh=4.0)
        assert bp.weight_inh == pytest.approx(2.0)

    def test_weight_inh_identity(self) -> None:
        """The inhibitory weight is g times the excitatory weight."""
        bp = BrunelParams()
        assert bp.weight_inh == bp.g_inh * bp.weight_exc
