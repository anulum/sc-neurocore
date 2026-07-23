# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLiquidTimeConstant from former test_model_ltc.py

"""Focused suite: TestLiquidTimeConstant from former test_model_ltc.py."""

from __future__ import annotations

from tests.model_ltc_support import *  # noqa: F403

class TestLiquidTimeConstant:
    def test_dynamics(self):
        from sc_neurocore.neurons.models.ltc import LiquidTimeConstantNeuron

        n = LiquidTimeConstantNeuron()
        for _ in range(50):
            n.step(2.0)
        assert n.x != 0.0
