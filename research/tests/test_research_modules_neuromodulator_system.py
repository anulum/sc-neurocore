# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuromodulatorSystem from former test_research_modules.py

"""Focused suite: TestNeuromodulatorSystem from former test_research_modules.py."""

from __future__ import annotations

from sc_neurocore.bio.neuromodulation import NeuromodulatorSystem


class TestNeuromodulatorSystem:
    """The dopamine, serotonin and norepinephrine neuromodulator model."""

    def test_defaults(self) -> None:
        """The default levels are 0.5, 0.5 and 0.1."""
        nm = NeuromodulatorSystem()
        assert nm.da_level == 0.5
        assert nm.ht_level == 0.5
        assert nm.ne_level == 0.1

    def test_reward_increases_dopamine(self) -> None:
        """A reward raises dopamine above 0.3."""
        nm = NeuromodulatorSystem(da_level=0.3)
        nm.update_levels(reward=1.0, stress=0.0)
        assert nm.da_level > 0.3

    def test_stress_increases_norepinephrine(self) -> None:
        """Stress raises norepinephrine above its default."""
        nm = NeuromodulatorSystem(ne_level=0.1)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ne_level > 0.1

    def test_stress_decreases_serotonin(self) -> None:
        """Stress lowers serotonin below 0.8."""
        nm = NeuromodulatorSystem(ht_level=0.8)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level < 0.8

    def test_serotonin_clipped(self) -> None:
        """Serotonin is clipped at its 0.1 minimum."""
        nm = NeuromodulatorSystem(ht_level=0.2)
        nm.update_levels(reward=0.0, stress=10.0)
        assert nm.ht_level >= 0.1  # clipped to min

    def test_modulate_neuron(self) -> None:
        """Modulation lowers the neuron threshold and adds a noise level."""
        nm = NeuromodulatorSystem(da_level=0.8, ht_level=0.6, ne_level=0.3)
        params = {"v_threshold": 1.0, "noise_std": 0.5}
        mod = nm.modulate_neuron(params)
        assert mod["v_threshold"] < 1.0  # DA lowers threshold
        assert "noise_std" in mod
