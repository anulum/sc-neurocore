# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLargeScalePopulation from former test_qc_e2e.py

"""Focused suite: TestLargeScalePopulation from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403

class TestLargeScalePopulation:
    """256 neurons, 1000 steps — no NaN, no Inf, memory bounded."""

    def test_256_neurons_1000_steps(self) -> None:
        """Stress test with large population."""
        pool = SpinPoolMPS(n_sites=256, bond_dim=16)
        neurons = [HybridFisherPosnerLIF(i, pool) for i in range(256)]

        rng = np.random.default_rng(42)
        total_spikes = 0
        for step in range(1000):
            for neuron in neurons:
                current = rng.normal(20.0, 10.0)
                _, spiked = neuron.step(current)
                if spiked:
                    total_spikes += 1

        # Verify no NaN/Inf in entanglement map
        assert np.all(np.isfinite(pool.entanglement_map)), "NaN/Inf in entanglement map"
        assert abs(np.sum(pool.entanglement_map) - 1.0) < 1e-8, "Normalisation drift"

        # Verify all neurons have finite ATP
        for i, n in enumerate(neurons):
            assert np.isfinite(n.atp_level), f"Neuron {i} ATP is NaN/Inf"
            assert 0.0 <= n.atp_level <= 1.0, f"Neuron {i} ATP={n.atp_level}"

        assert total_spikes > 0, "Should produce spikes in 1000 steps"
