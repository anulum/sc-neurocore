# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAIOptimizedNetwork from former test_model_ai_optimized.py

"""Focused suite: TestAIOptimizedNetwork from former test_model_ai_optimized.py."""

from __future__ import annotations

from tests.model_ai_optimized_support import *  # noqa: F403

class TestAIOptimizedNetwork:
    """Pipeline wiring for representative models."""

    def test_multitimescale_network(self):
        pop = Population(MultiTimescaleNeuron, n=5, label="mt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_differentiable_surrogate_network(self):
        pop = Population(DifferentiableSurrogateNeuron, n=10, label="ds")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_compositional_binding_network(self):
        pop = Population(CompositionalBindingNeuron, n=10, label="cb")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0
