# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHillTononi from former test_model_hill_tononi.py

"""Focused suite: TestHillTononi from former test_model_hill_tononi.py."""

from __future__ import annotations

from tests.model_hill_tononi_support import *  # noqa: F403


class TestHillTononi:
    def test_fires(self):
        from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

        n = HillTononiNeuron()
        assert sum(n.step(5.0) for _ in range(300)) > 0
