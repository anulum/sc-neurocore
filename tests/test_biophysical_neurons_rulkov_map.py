# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRulkovMap from former test_biophysical_neurons.py

"""Focused suite: TestRulkovMap from former test_biophysical_neurons.py."""

from __future__ import annotations


class TestRulkovMap:
    def test_fires(self) -> None:
        from sc_neurocore.neurons.models import RulkovMapNeuron

        n = RulkovMapNeuron(alpha=6.0, sigma=0.0)
        spikes = sum(n.step(0.1) for _ in range(2000))
        assert spikes > 0

    def test_deterministic(self) -> None:
        from sc_neurocore.neurons.models import RulkovMapNeuron

        n1 = RulkovMapNeuron()
        n2 = RulkovMapNeuron()
        s1 = [n1.step() for _ in range(100)]
        s2 = [n2.step() for _ in range(100)]
        assert s1 == s2
