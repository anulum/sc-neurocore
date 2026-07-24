# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Parameters from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Parameters from former test_model_plant_r15.py."""

from __future__ import annotations

from tests.model_plant_r15_support import *  # noqa: F403


class TestPlantR15Parameters:
    def test_g_kca_controls_burst_termination(self):
        """Reducing g_KCa should allow more spikes (less Ca-K inhibition)."""
        n_low = PlantR15Neuron(g_kca=0.001)
        n_high = PlantR15Neuron(g_kca=0.03)
        s_low, _ = _run(n_low, current=0.0, steps=50000)
        s_high, _ = _run(n_high, current=0.0, steps=50000)
        assert len(s_low) >= len(s_high), f"Low g_KCa: {len(s_low)} spikes, high: {len(s_high)}"

    def test_tau_ca_affects_calcium_dynamics(self):
        """Shorter tau_Ca → faster Ca decay → different equilibrium."""
        n_fast = PlantR15Neuron(tau_ca=100.0)
        n_slow = PlantR15Neuron(tau_ca=1000.0)
        for _ in range(50000):
            n_fast.step(0.0)
            n_slow.step(0.0)
        # Faster decay → lower steady-state Ca
        assert n_fast.ca < n_slow.ca, f"Fast Ca={n_fast.ca:.4f}, slow Ca={n_slow.ca:.4f}"
