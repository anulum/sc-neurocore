# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuromodulatorSystem from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestNeuromodulatorSystem from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

from tests.bio_chaos_spatial_learning_support import *  # noqa: F403

class TestNeuromodulatorSystem:
    def test_update_levels(self):
        nm = NeuromodulatorSystem()
        nm.update_levels(reward=1.0, stress=0.8)
        assert nm.da_level != 0.5
        assert nm.ne_level != 0.1
        assert 0.1 <= nm.ht_level <= 1.0

    def test_modulate_neuron_all_keys(self):
        nm = NeuromodulatorSystem(da_level=0.8, ht_level=0.6, ne_level=0.3)
        params = {"v_threshold": 1.0, "noise_std": 0.5}
        mod = nm.modulate_neuron(params)
        assert mod["v_threshold"] < 1.0
        assert mod["noise_std"] != 0.5

    def test_modulate_neuron_no_keys(self):
        nm = NeuromodulatorSystem()
        params = {"tau_mem": 10.0}
        assert nm.modulate_neuron(params) == {"tau_mem": 10.0}

    def test_serotonin_clip(self):
        nm = NeuromodulatorSystem(ht_level=0.15)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level >= 0.1
