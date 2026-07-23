# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVIPACurrent from former test_model_vip_neuron.py

"""Focused suite: TestVIPACurrent from former test_model_vip_neuron.py."""

from __future__ import annotations

from tests.model_vip_neuron_support import *  # noqa: F403

class TestVIPACurrent:
    def test_a_current_block_changes_firing(self):
        intact = _spikes(VIPNeuron(), 1.0, 40000)
        blocked = _spikes(VIPNeuron(g_a=0.0), 1.0, 40000)
        assert intact != blocked

    def test_a_gate_inactivates_during_drive(self):
        # The b inactivation gate falls from its rested 0.9 under sustained drive.
        n = VIPNeuron()
        for _ in range(2000):
            n.step(1.0)
        assert n.b < 0.9
