# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuromodulatorSystem from former test_research_modules.py

"""Focused suite: TestNeuromodulatorSystem from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestNeuromodulatorSystem:
    def test_defaults(self):
        nm = NeuromodulatorSystem()
        assert nm.da_level == 0.5
        assert nm.ht_level == 0.5
        assert nm.ne_level == 0.1

    def test_reward_increases_dopamine(self):
        nm = NeuromodulatorSystem(da_level=0.3)
        nm.update_levels(reward=1.0, stress=0.0)
        assert nm.da_level > 0.3

    def test_stress_increases_norepinephrine(self):
        nm = NeuromodulatorSystem(ne_level=0.1)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ne_level > 0.1

    def test_stress_decreases_serotonin(self):
        nm = NeuromodulatorSystem(ht_level=0.8)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level < 0.8

    def test_serotonin_clipped(self):
        nm = NeuromodulatorSystem(ht_level=0.2)
        nm.update_levels(reward=0.0, stress=10.0)
        assert nm.ht_level >= 0.1  # clipped to min

    def test_modulate_neuron(self):
        nm = NeuromodulatorSystem(da_level=0.8, ht_level=0.6, ne_level=0.3)
        params = {"v_threshold": 1.0, "noise_std": 0.5}
        mod = nm.modulate_neuron(params)
        assert mod["v_threshold"] < 1.0  # DA lowers threshold
        assert "noise_std" in mod
