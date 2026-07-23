# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelCompartments from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelCompartments from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403

class TestPinskyRinzelCompartments:
    def test_soma_dendrite_coupling(self):
        coupled = PinskyRinzelNeuron(gc=2.1)
        uncoupled = PinskyRinzelNeuron(gc=0.001)
        for _ in range(5000):
            coupled.step(20.0, 0.0)
            uncoupled.step(20.0, 0.0)
        assert abs(coupled.v_d - uncoupled.v_d) > 1.0

    def test_dendritic_drive_evokes_spiking(self):
        n = PinskyRinzelNeuron()
        assert len(_run(n, current_soma=0.0, steps=50000, current_dend=20.0)) > 0

    def test_calcium_accumulates_during_spiking(self):
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(20.0)
        assert n.ca > 1.0

    def test_calcium_non_negative_without_drive(self):
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(0.0)
        assert n.ca >= 0.0

    def test_gc_coupling_strength_reduces_gap(self):
        weak = PinskyRinzelNeuron(gc=0.5)
        strong = PinskyRinzelNeuron(gc=5.0)
        for _ in range(10000):
            weak.step(20.0)
            strong.step(20.0)
        assert abs(strong.v_s - strong.v_d) < abs(weak.v_s - weak.v_d)
