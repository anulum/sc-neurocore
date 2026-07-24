# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSTAdaptation from former test_model_sst_neuron.py

"""Focused suite: TestSSTAdaptation from former test_model_sst_neuron.py."""

from __future__ import annotations

from tests.model_sst_neuron_support import *  # noqa: F403


class TestSSTAdaptation:
    def test_m_current_block_changes_firing(self):
        intact = _spikes(SSTNeuron(), 2.0, 40000)
        blocked = _spikes(SSTNeuron(g_m=0.0), 2.0, 40000)
        assert intact != blocked
