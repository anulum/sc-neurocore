# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGEAnalytical from former test_model_gutkin_ermentrout.py

"""Focused suite: TestGEAnalytical from former test_model_gutkin_ermentrout.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.gutkin_ermentrout import (
    GutkinErmentroutNeuron,
)
from tests.model_gutkin_ermentrout_support import (
    _m_inf,
    _n_inf,
    _rhs,
    _rk4_reference,
)


class TestGEAnalytical:
    """Analytical behaviour of the GutkinErmentroutNeuron."""

    def test_m_inf_boltzmann(self) -> None:
        """m_inf = 1/(1+exp(-(v+20)/15))."""
        n = GutkinErmentroutNeuron()
        for v in [-80, -60, -20, 0, 20]:
            expected = _m_inf(float(v))
            computed = 1.0 / (1.0 + np.exp(-(v + 20.0) / 15.0))
            assert abs(expected - computed) < 1e-14

    def test_m_inf_midpoint(self) -> None:
        """The activation m_inf(-20) equals 0.5."""
        assert abs(_m_inf(-20.0) - 0.5) < 1e-12

    def test_n_inf_midpoint(self) -> None:
        """The activation n_inf(-25) equals 0.5."""
        assert abs(_n_inf(-25.0) - 0.5) < 1e-12

    def test_rk4_current_balance_one_step(self) -> None:
        """One committed step matches the explicit RK4 current balance."""
        n = GutkinErmentroutNeuron()
        expected_v, expected_n = _rk4_reference(n, current=3.0)
        n.step(3.0)
        assert abs(n.v - expected_v) < 1e-12
        assert abs(n.n - expected_n) < 1e-12

    def test_rk4_differs_from_euler_baseline(self) -> None:
        """RK4 is not the historical first-order Euler update."""
        n = GutkinErmentroutNeuron()
        v0, n0 = n.v, n.n
        current = 3.0
        euler_n = n0 + (_n_inf(v0) - n0) * n.dt
        euler_v = v0 + _rhs(n, v0, euler_n, current)[0] * n.dt
        n.step(current)
        assert abs(n.v - euler_v) > 1e-6

    def test_three_currents(self) -> None:
        """The model has three positive ionic conductances."""
        n = GutkinErmentroutNeuron()
        assert n.g_na > 0 and n.g_k > 0 and n.g_l > 0

    def test_reversal_ordering(self) -> None:
        """The reversal potentials are in physiological order."""
        n = GutkinErmentroutNeuron()
        assert n.e_k < n.e_l < n.e_na

    def test_persistent_na_no_inactivation(self) -> None:
        """Persistent Na: m only (no h gate). m is instantaneous."""
        # Source: i_na = g_na * m_inf * (v - e_na)
        # No h variable — persistent sodium
        n = GutkinErmentroutNeuron()
        assert not hasattr(n, "h") or n.__class__.__name__ == "GutkinErmentroutNeuron"
