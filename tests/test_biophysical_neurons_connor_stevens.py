# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConnorStevens from former test_biophysical_neurons.py

"""Focused suite: TestConnorStevens from former test_biophysical_neurons.py."""

from __future__ import annotations

from tests.biophysical_neurons_support import *  # noqa: F403

class TestConnorStevens:
    def test_fires(self):
        from sc_neurocore.neurons.models import ConnorStevensNeuron

        n = ConnorStevensNeuron()
        spikes = sum(n.step(10.0) for _ in range(100))
        assert spikes > 0

    def test_a_type_current(self):
        from sc_neurocore.neurons.models import ConnorStevensNeuron

        n = ConnorStevensNeuron()
        for _ in range(50):
            n.step(8.0)
        assert n.a != 0.5, "A-type activation must evolve"
