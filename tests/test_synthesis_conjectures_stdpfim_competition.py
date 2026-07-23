# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTDPFIMCompetition from former test_synthesis_conjectures.py

"""Focused suite: TestSTDPFIMCompetition from former test_synthesis_conjectures.py."""

from __future__ import annotations

from tests.synthesis_conjectures_support import *  # noqa: F403

class TestSTDPFIMCompetition:
    """STDP asymmetric updates and FIM symmetric corrections should
    produce measurably different weight trajectories."""

    def test_stdp_only_vs_stdp_plus_fim(self):
        """With FIM active, weight distribution should differ from STDP-only."""
        results = {}
        for label, lam in [("stdp_only", 0.0), ("stdp_fim", 5.0)]:
            pop = Population(StochasticLIFNeuron, n=20, label="e")
            proj = Projection(pop, pop, weight=0.3, probability=0.3, plasticity="stdp", seed=42)
            drive = PoissonInput(n=20, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
            net = Network(pop, proj, drive, fim_lambda=lam)
            net.run(duration=0.2, dt=0.001)
            results[label] = proj.data.copy()

        # Weight distributions should differ
        diff = np.mean(np.abs(results["stdp_only"] - results["stdp_fim"]))
        assert diff > 0.0001, f"FIM had no measurable effect (diff={diff:.6f})"

    def test_fim_does_not_collapse_weights(self):
        """FIM should not drive all weights to zero or a single value."""
        pop = Population(StochasticLIFNeuron, n=20, label="e")
        proj = Projection(pop, pop, weight=0.3, probability=0.3, plasticity="stdp", seed=42)
        drive = PoissonInput(n=20, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        net = Network(pop, proj, drive, fim_lambda=10.0)
        net.run(duration=0.3, dt=0.001)
        # Weights should have nonzero variance
        assert np.std(proj.data) > 0.001, "FIM collapsed all weights"
