# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWendlingFourPopulations from former test_model_wendling.py

"""Focused suite: TestWendlingFourPopulations from former test_model_wendling.py."""

from __future__ import annotations

from tests.model_wendling_support import *  # noqa: F403


class TestWendlingFourPopulations:
    """4 populations: pyramidal (y0), excitatory (y1), fast inh (y2), slow inh (y3)."""

    def test_excitatory_gains(self):
        """a_exc controls excitatory PSP amplitude."""
        n_weak = WendlingNeuron(a_exc=1.0)
        n_strong = WendlingNeuron(a_exc=5.0)
        for _ in range(10000):
            n_weak.step(220.0)
            n_strong.step(220.0)
        assert abs(n_weak.y1) != abs(n_strong.y1)

    def test_fast_inhibition(self):
        """b_fast controls fast GABA_A inhibition amplitude."""
        n_weak = WendlingNeuron(b_fast=10.0)
        n_strong = WendlingNeuron(b_fast=40.0)
        for _ in range(10000):
            n_weak.step(220.0)
            n_strong.step(220.0)
        assert abs(n_weak.y2) != abs(n_strong.y2)

    def test_slow_inhibition(self):
        """g_slow controls slow GABA_B inhibition."""
        n_weak = WendlingNeuron(g_slow=5.0)
        n_strong = WendlingNeuron(g_slow=20.0)
        for _ in range(10000):
            n_weak.step(220.0)
            n_strong.step(220.0)
        assert abs(n_weak.y3) != abs(n_strong.y3)
