# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuromodulatorSystem from former test_bio.py

"""Focused suite: TestNeuromodulatorSystem from former test_bio.py."""

from __future__ import annotations

from sc_neurocore.bio import (
    NeuromodulatorSystem,
)


class TestNeuromodulatorSystem:
    """Neuromodulator responses to reward and stress."""

    def test_defaults(self) -> None:
        """Levels start at their defaults."""
        nm = NeuromodulatorSystem()
        assert nm.da_level == 0.5
        assert nm.ht_level == 0.5
        assert nm.ne_level == 0.1

    def test_reward_boosts_dopamine(self) -> None:
        """Reward raises dopamine."""
        nm = NeuromodulatorSystem(da_level=0.5)
        nm.update_levels(reward=1.0, stress=0.0)
        assert nm.da_level > 0.5

    def test_stress_boosts_ne(self) -> None:
        """Stress raises noradrenaline."""
        nm = NeuromodulatorSystem(ne_level=0.1)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ne_level > 0.1

    def test_stress_drops_serotonin(self) -> None:
        """Stress lowers serotonin."""
        nm = NeuromodulatorSystem(ht_level=0.5)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level < 0.5

    def test_serotonin_bounded(self) -> None:
        """Serotonin never falls below 0.1."""
        nm = NeuromodulatorSystem()
        for _ in range(100):
            nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level >= 0.1

    def test_modulate_neuron(self) -> None:
        """Modulation lowers the threshold and sets noise."""
        nm = NeuromodulatorSystem(da_level=0.8, ht_level=0.5, ne_level=0.3)
        params = {"v_threshold": 1.0, "noise_std": 0.1}
        mod = nm.modulate_neuron(params)
        assert mod["v_threshold"] < 1.0
        assert "noise_std" in mod
