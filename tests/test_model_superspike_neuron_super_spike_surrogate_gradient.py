# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSuperSpikeSurrogateGradient from former test_model_superspike_neuron.py

"""Focused suite: TestSuperSpikeSurrogateGradient from former test_model_superspike_neuron.py."""

from __future__ import annotations

from tests.model_superspike_neuron_support import *  # noqa: F403


class TestSuperSpikeSurrogateGradient:
    """Core: σ'(V) = 1/(β|V-θ|+1)². Peaks at V=θ, decays with distance."""

    def test_sg_peak_at_threshold(self):
        """σ'(θ) = 1/(0+1)² = 1.0 — maximum at threshold."""
        n = SuperSpikeNeuron()
        n.v = n.v_threshold
        assert abs(n.surrogate_grad() - 1.0) < 1e-10

    def test_sg_symmetric_around_threshold(self):
        """σ'(θ+δ) = σ'(θ-δ) — symmetric in |V-θ|."""
        n = SuperSpikeNeuron()
        for delta in [0.1, 0.5, 1.0, 5.0]:
            n.v = n.v_threshold + delta
            sg_above = n.surrogate_grad()
            n.v = n.v_threshold - delta
            sg_below = n.surrogate_grad()
            assert abs(sg_above - sg_below) < 1e-10, f"delta={delta}"

    def test_sg_decays_with_distance(self):
        """σ' decreases as |V-θ| increases."""
        n = SuperSpikeNeuron()
        sgs = []
        for v in [1.0, 0.9, 0.5, 0.0, -1.0]:
            n.v = v
            sgs.append(n.surrogate_grad())
        # Should be monotonically decreasing
        assert all(sgs[i] >= sgs[i + 1] for i in range(len(sgs) - 1))

    def test_sg_formula_exact(self):
        """Verify σ' = 1/(β|V-θ|+1)² at specific V."""
        n = SuperSpikeNeuron()
        n.v = 0.5  # |V-θ| = 0.5
        expected = 1.0 / (n.beta_sg * 0.5 + 1.0) ** 2
        assert abs(n.surrogate_grad() - expected) < 1e-10

    def test_beta_controls_sharpness(self):
        """Higher beta → sharper peak (faster decay away from θ)."""
        n_sharp = SuperSpikeNeuron(beta_sg=50.0)
        n_soft = SuperSpikeNeuron(beta_sg=1.0)
        # At V = θ - 0.5:
        n_sharp.v = 0.5
        n_soft.v = 0.5
        assert n_soft.surrogate_grad() > n_sharp.surrogate_grad()
