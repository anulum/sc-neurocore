# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeSchutterPurkinje from former test_model_de_schutter_purkinje.py

"""Focused suite: TestDeSchutterPurkinje from former test_model_de_schutter_purkinje.py."""

from __future__ import annotations

from tests.model_de_schutter_purkinje_support import *  # noqa: F403


class TestDeSchutterPurkinje:
    def test_dynamics(self) -> None:
        from sc_neurocore.neurons.models.de_schutter_purkinje import DeSchutterPurkinjeNeuron

        n = DeSchutterPurkinjeNeuron()
        for _ in range(200):
            n.step(20.0)
        assert n.ca != 0.0001, "calcium must evolve"

    def test_gating_bounded(self) -> None:
        from sc_neurocore.neurons.models.de_schutter_purkinje import DeSchutterPurkinjeNeuron

        n = DeSchutterPurkinjeNeuron()
        for _ in range(100):
            n.step(15.0)
        assert 0.0 <= n.h_na <= 1.0
        assert 0.0 <= n.n_k <= 1.0
