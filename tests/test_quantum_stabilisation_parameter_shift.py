# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParameterShift from former test_quantum_stabilisation.py

"""Focused suite: TestParameterShift from former test_quantum_stabilisation.py."""

from __future__ import annotations

from typing import Any
import numpy as np


class TestParameterShift:
    """The parameter-shift gradient rule."""

    def test_sin_gradient(self) -> None:
        """The shift rule gives the exact gradient of a sine."""
        from sc_neurocore.quantum.param_shift import parameter_shift_gradient

        def f(p: np.ndarray[Any, Any]) -> float:
            return float(np.sin(p[0]))

        params = np.array([0.5])
        grad = parameter_shift_gradient(f, params)
        np.testing.assert_allclose(grad[0], np.cos(0.5), atol=1e-10)

    def test_multivariate(self) -> None:
        """The shift rule gives each partial derivative."""
        from sc_neurocore.quantum.param_shift import parameter_shift_gradient

        def f(p: np.ndarray[Any, Any]) -> float:
            return float(np.sin(p[0]) + np.cos(p[1]))

        params = np.array([1.0, 2.0])
        grad = parameter_shift_gradient(f, params)
        np.testing.assert_allclose(grad[0], np.cos(1.0), atol=1e-10)
        np.testing.assert_allclose(grad[1], -np.sin(2.0), atol=1e-10)

    def test_optimizer_converges(self) -> None:
        """The shift-rule optimiser converges to the minimum."""
        from sc_neurocore.quantum.param_shift import ParameterShiftOptimizer

        def f(p: np.ndarray[Any, Any]) -> float:
            return float((p[0] - 1.0) ** 2)

        opt = ParameterShiftOptimizer(f, 1, lr=0.1)
        params = np.array([0.0])
        for _ in range(50):
            params = opt.step(params)
        assert abs(params[0] - 1.0) < 0.1
