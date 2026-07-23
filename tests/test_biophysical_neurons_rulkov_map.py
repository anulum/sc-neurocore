# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRulkovMap from former test_biophysical_neurons.py

"""Focused suite: TestRulkovMap from former test_biophysical_neurons.py."""

from __future__ import annotations

from tests.biophysical_neurons_support import *  # noqa: F403

class TestRulkovMap:
    def test_fires(self):
        from sc_neurocore.neurons.models import RulkovMapNeuron

        n = RulkovMapNeuron(alpha=6.0, sigma=0.0, x_threshold=-0.5)
        spikes = sum(n.step(0.1) for _ in range(2000))
        assert spikes > 0

    def test_deterministic(self):
        from sc_neurocore.neurons.models import RulkovMapNeuron

        n1 = RulkovMapNeuron()
        n2 = RulkovMapNeuron()
        s1 = [n1.step() for _ in range(100)]
        s2 = [n2.step() for _ in range(100)]
        assert s1 == s2
