# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLearnableNeuronModel from former test_model_lnm.py

"""Focused suite: TestLearnableNeuronModel from former test_model_lnm.py."""

from __future__ import annotations

from tests.model_lnm_support import *  # noqa: F403


class TestLearnableNeuronModel:
    def test_fires(self):
        from sc_neurocore.neurons.models.lnm import LearnableNeuronModel

        n = LearnableNeuronModel()
        assert sum(n.step(2.0) for _ in range(50)) > 0
