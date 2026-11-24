# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertErfApprox from former test_model_siegert.py

"""Focused suite: TestSiegertErfApprox from former test_model_siegert.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.siegert import (
    _erf_approx,
)
from scipy.special import (
    erf as scipy_erf,
)


class TestSiegertErfApprox:
    """Abramowitz & Stegun 7.1.26 rational approximation."""

    def test_erf_at_zero(self) -> None:
        """The error-function approximation is zero at zero."""
        result = _erf_approx(np.array([0.0]))
        assert abs(result[0]) < 1e-6

    def test_erf_symmetry(self) -> None:
        """The error-function approximation is odd: erf(-x) = -erf(x)."""
        x = np.array([1.0, -1.0])
        result = _erf_approx(x)
        assert abs(result[0] + result[1]) < 1e-6

    def test_erf_accuracy(self) -> None:
        """Compare directly with the maintained SciPy reference."""
        x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        approx = _erf_approx(x)
        exact = scipy_erf(x)
        max_err = np.max(np.abs(approx - exact))
        assert max_err < 1e-6, f"max erf error = {max_err:.2e}"

    def test_erf_bounded(self) -> None:
        """The error-function approximation stays inside [-1, 1]."""
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = _erf_approx(x)
        assert np.all(np.abs(result) <= 1.0 + 1e-6)
