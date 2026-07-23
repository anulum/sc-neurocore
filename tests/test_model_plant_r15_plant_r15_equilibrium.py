# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Equilibrium from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Equilibrium from former test_model_plant_r15.py."""

from __future__ import annotations

from tests.model_plant_r15_support import *  # noqa: F403

class TestPlantR15Equilibrium:
    def test_transient_spike(self):
        """From initial conditions, model fires exactly 1 transient spike."""
        n = PlantR15Neuron()
        spike_times, _ = _run(n, current=0.0, steps=100000)
        assert len(spike_times) == 1, f"Expected 1 transient spike, got {len(spike_times)}"

    def test_converges_to_fixed_point(self):
        """After transient, V stabilises near −23.8 mV (equilibrium)."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        v_eq = n.v
        # Run 10k more steps — V should barely change
        for _ in range(10000):
            n.step(0.0)
        assert abs(n.v - v_eq) < 0.01, (
            f"V drifted from {v_eq:.3f} to {n.v:.3f} — not at equilibrium"
        )
        assert -30.0 < v_eq < -15.0, f"V_eq = {v_eq:.2f} outside expected range"

    def test_equilibrium_independent_of_small_current(self):
        """Small currents (I<1) shift equilibrium slightly but don't trigger
        sustained oscillation — model stays at a (shifted) fixed point."""
        for I in [0.0, 0.1, 0.5]:
            n = PlantR15Neuron()
            spike_times, _ = _run(n, current=I, steps=100000)
            assert len(spike_times) <= 2, (
                f"I={I}: {len(spike_times)} spikes — expected ≤2 (transient only)"
            )
