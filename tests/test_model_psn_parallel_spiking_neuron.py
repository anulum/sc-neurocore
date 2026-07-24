# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParallelSpikingNeuron from former test_model_psn.py

"""Focused suite: TestParallelSpikingNeuron from former test_model_psn.py."""

from __future__ import annotations

from tests.model_psn_support import *  # noqa: F403


class TestParallelSpikingNeuron:
    def test_dynamics(self):
        from sc_neurocore.neurons.models.psn import ParallelSpikingNeuron

        n = ParallelSpikingNeuron()
        results = [n.step(0.3) for _ in range(20)]
        assert any(r != 0 for r in results) or any(b != 0.0 for b in n.buffer)
