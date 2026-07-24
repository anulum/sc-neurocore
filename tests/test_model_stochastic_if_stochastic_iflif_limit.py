# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticIFLIFLimit from former test_model_stochastic_if.py

"""Focused suite: TestStochasticIFLIFLimit from former test_model_stochastic_if.py."""

from __future__ import annotations

from tests.model_stochastic_if_support import *  # noqa: F403


class TestStochasticIFLIFLimit:
    """At sigma=0, model reduces to standard LIF. Verify LIF properties."""

    def test_lif_membrane_equation(self):
        """dV/dt = (-(V-V_rest) + I) / tau_m. Verify one step."""
        n = StochasticIFNeuron(sigma=0.0, mu=0.0)
        v0 = n.v
        I = 15.0
        n.step(I)
        expected = v0 + (-(v0 - n.v_rest) + I) / n.tau_m * n.dt
        assert abs(n.v - expected) < 1e-10

    def test_lif_steady_state(self):
        """V_ss = V_rest + mu + I. At I=15, sigma=0: V_ss = -55 < threshold."""
        n = StochasticIFNeuron(sigma=0.0, mu=0.0)
        for _ in range(10000):
            n.step(15.0)
        expected_vss = n.v_rest + 15.0  # -55
        assert abs(n.v - expected_vss) < 0.1
