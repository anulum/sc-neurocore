# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOmegaIntegrator from former test_zero_coverage_a.py

"""Focused suite: TestOmegaIntegrator from former test_zero_coverage_a.py."""

from __future__ import annotations

import numpy as np


class TestOmegaIntegrator:
    """The omega integrator research module."""

    def test_unify(self) -> None:
        """Unification returns an array (the check is on the type only)."""
        from meta.omega import OmegaIntegrator

        o = OmegaIntegrator()
        states = [np.random.randn(4), np.random.randn(4)]
        result = o.unify(states)
        assert isinstance(result, np.ndarray)
