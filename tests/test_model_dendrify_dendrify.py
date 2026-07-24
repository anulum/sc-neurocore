# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDendrify from former test_model_dendrify.py

"""Focused suite: TestDendrify from former test_model_dendrify.py."""

from __future__ import annotations

from tests.model_dendrify_support import *  # noqa: F403


class TestDendrify:
    def test_dynamics(self):
        from sc_neurocore.neurons.models.dendrify import DendrifyNeuron

        n = DendrifyNeuron()
        for _ in range(200):
            n.step(10.0)
        assert n.v_s != -65.0 or n.v_d != -65.0
