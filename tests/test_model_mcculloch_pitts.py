# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: McCullochPittsNeuron

"""Full pipeline test for McCullochPittsNeuron (McCulloch & Pitts 1943).

The first mathematical neuron: y = 1 if input >= theta, else 0.
Stateless, deterministic, binary."""

from __future__ import annotations

from sc_neurocore.neurons.models.mcculloch_pitts import McCullochPittsNeuron
from sc_neurocore.network.population import Population


class TestMPIsolation:
    def test_construction(self):
        n = McCullochPittsNeuron()
        assert n.theta == 1.0

    def test_step_returns_binary(self):
        assert McCullochPittsNeuron().step(0.0) in (0, 1)

    def test_below_threshold(self):
        n = McCullochPittsNeuron()
        assert n.step(0.5) == 0

    def test_at_threshold(self):
        n = McCullochPittsNeuron()
        assert n.step(1.0) == 1

    def test_above_threshold(self):
        n = McCullochPittsNeuron()
        assert n.step(5.0) == 1

    def test_negative_input(self):
        n = McCullochPittsNeuron()
        assert n.step(-1.0) == 0

    def test_stateless(self):
        """Same input always gives same output regardless of history."""
        n = McCullochPittsNeuron()
        n.step(5.0)
        n.step(5.0)
        assert n.step(0.5) == 0

    def test_custom_theta(self):
        n = McCullochPittsNeuron(theta=0.5)
        assert n.step(0.5) == 1
        assert n.step(0.4) == 0

    def test_reset_noop(self):
        n = McCullochPittsNeuron()
        n.reset()
        assert n.theta == 1.0

    def test_deterministic(self):
        n1 = McCullochPittsNeuron()
        n2 = McCullochPittsNeuron()
        for i in range(100):
            assert n1.step(float(i) / 50.0) == n2.step(float(i) / 50.0)

    def test_logic_and(self):
        """Two inputs with theta=2 implements AND gate."""
        n = McCullochPittsNeuron(theta=2.0)
        assert n.step(0.0 + 0.0) == 0
        assert n.step(1.0 + 0.0) == 0
        assert n.step(0.0 + 1.0) == 0
        assert n.step(1.0 + 1.0) == 1

    def test_logic_or(self):
        """Two inputs with theta=1 implements OR gate."""
        n = McCullochPittsNeuron(theta=1.0)
        assert n.step(0.0 + 0.0) == 0
        assert n.step(1.0 + 0.0) == 1
        assert n.step(0.0 + 1.0) == 1
        assert n.step(1.0 + 1.0) == 1


class TestMPNetwork:
    def test_population(self):
        assert Population(McCullochPittsNeuron, n=10, label="mp").n == 10
