# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LarterBreakspearNeuron

"""Full pipeline test for LarterBreakspearNeuron (Breakspear et al. 2003).

Neural mass with ion channels (Ca, Na, K, leak). 3 ODEs per node.
Returns continuous voltage, not binary spikes. Used in TVB."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.larter_breakspear import LarterBreakspearNeuron
from sc_neurocore.network.population import Population


class TestLBIsolation:
    def test_construction(self):
        n = LarterBreakspearNeuron()
        assert n.v == -0.5
        assert n.g_ca == 1.1

    def test_step_returns_float(self):
        n = LarterBreakspearNeuron()
        v = n.step()
        assert isinstance(v, (float, np.floating))

    def test_oscillates(self):
        """Default params should produce oscillatory dynamics."""
        n = LarterBreakspearNeuron()
        vals = [n.step() for _ in range(10000)]
        assert np.std(vals) > 0.01

    def test_bounded(self):
        n = LarterBreakspearNeuron()
        vals = [n.step() for _ in range(10000)]
        assert max(vals) < 5.0
        assert min(vals) > -5.0

    def test_three_state_variables(self):
        n = LarterBreakspearNeuron()
        for _ in range(2000):
            n.step()
        assert n.v != -0.5
        assert n.w != 0.0

    def test_coupling_input(self):
        """External coupling should shift dynamics."""
        n_no = LarterBreakspearNeuron()
        n_yes = LarterBreakspearNeuron()
        for _ in range(5000):
            n_no.step(0.0)
            n_yes.step(0.5)
        assert n_no.v != n_yes.v

    def test_sigmoid_gates(self):
        n = LarterBreakspearNeuron()
        assert 0.0 < n._m_ca(0.0) < 1.0
        assert 0.0 < n._m_na(0.0) < 1.0
        assert 0.0 < n._m_k(0.0) < 1.0

    def test_numerical_stability(self):
        for coupling in [0.0, 0.3, 1.0]:
            n = LarterBreakspearNeuron()
            for _ in range(10000):
                n.step(coupling)
            assert np.isfinite(n.v), f"v NaN at c={coupling}"
            assert np.isfinite(n.w), f"w NaN at c={coupling}"
            assert np.isfinite(n.z), f"z NaN at c={coupling}"

    def test_reset(self):
        n = LarterBreakspearNeuron()
        for _ in range(3000):
            n.step()
        n.reset()
        assert n.v == -0.5
        assert n.w == 0.0
        assert n.z == 0.0

    def test_deterministic(self):
        n1 = LarterBreakspearNeuron()
        n2 = LarterBreakspearNeuron()
        for _ in range(500):
            assert n1.step() == n2.step()


class TestLBNetwork:
    def test_population(self):
        assert Population(LarterBreakspearNeuron, n=5, label="lb").n == 5
