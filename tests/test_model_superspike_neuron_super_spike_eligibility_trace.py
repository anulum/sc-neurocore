# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSuperSpikeEligibilityTrace from former test_model_superspike_neuron.py

"""Focused suite: TestSuperSpikeEligibilityTrace from former test_model_superspike_neuron.py."""

from __future__ import annotations

from tests.model_superspike_neuron_support import *  # noqa: F403


class TestSuperSpikeEligibilityTrace:
    """trace = α_e · trace + σ'(V). Leaky integrator of surrogate gradient."""

    def test_trace_accumulates_sg(self):
        """Trace grows when σ'(V) > 0 (always, but peaks near threshold)."""
        n = SuperSpikeNeuron()
        t0 = n.trace
        n.step(0.5)
        assert n.trace > t0

    def test_trace_decays_without_sg(self):
        """With V far from threshold: σ' ≈ 0, trace decays."""
        n = SuperSpikeNeuron()
        n.trace = 5.0
        n.v = -10.0  # far from threshold → σ' ≈ 0
        n.step(0.0)  # v stays negative, σ' tiny
        assert n.trace < 5.0

    def test_trace_peaks_near_threshold(self):
        """When V hovers near threshold, trace accumulates fastest."""
        # Near threshold: higher SG → faster trace growth
        n_near = SuperSpikeNeuron()
        n_far = SuperSpikeNeuron()
        for _ in range(100):
            n_near.step(0.09)  # v ≈ 0.9, near threshold
            n_far.step(0.001)  # v ≈ 0.01, far from threshold
        assert n_near.trace > n_far.trace
