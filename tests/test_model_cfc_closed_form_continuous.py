# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestClosedFormContinuous from former test_model_cfc.py

"""Focused suite: TestClosedFormContinuous from former test_model_cfc.py."""

from __future__ import annotations

from tests.model_cfc_support import *  # noqa: F403

class TestClosedFormContinuous:
    def test_dynamics(self):
        from sc_neurocore.neurons.models.cfc import ClosedFormContinuousNeuron

        n = ClosedFormContinuousNeuron()
        for _ in range(20):
            n.step(1.0)
        assert n.x != 0.0
