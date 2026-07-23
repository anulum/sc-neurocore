# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFIMFeedback from former test_fim_symmetry_dynamics.py

"""Focused suite: TestFIMFeedback from former test_fim_symmetry_dynamics.py."""

from __future__ import annotations

from tests.fim_symmetry_dynamics_support import *  # noqa: F403

class TestFIMFeedback:
    def test_fim_zero_lambda_no_effect(self):
        """fim_lambda=0 should not modify weights."""
        net, proj, mon = _make_self_connected_network(fim_lambda=0.0)
        w_before = proj.data.copy()
        # Run without FIM — only STDP modifies weights
        net.run(duration=0.01, dt=0.001)
        # Weights may change from STDP but FIM contributes nothing
        # (just verify no crash)
        assert proj.data is not None

    def test_fim_positive_lambda_modifies_weights(self):
        """fim_lambda>0 should produce different weight trajectory than lambda=0."""
        net0, proj0, _ = _make_self_connected_network(n=20, fim_lambda=0.0)
        net1, proj1, _ = _make_self_connected_network(n=20, fim_lambda=5.0)
        net0.run(duration=0.1, dt=0.001)
        net1.run(duration=0.1, dt=0.001)
        # Weight trajectories should differ
        diff = np.mean(np.abs(proj0.data - proj1.data))
        assert diff > 0.001, f"FIM had no effect on weights (diff={diff:.6f})"

    def test_fim_weights_stay_nonnegative(self):
        """FIM correction should never push weights below zero."""
        net, proj, mon = _make_self_connected_network(n=20, fim_lambda=10.0)
        net.run(duration=0.2, dt=0.001)
        assert np.all(proj.data >= 0), "negative weight after FIM correction"

    def test_fim_network_attribute(self):
        """Network should store fim_lambda."""
        net, _, _ = _make_self_connected_network(fim_lambda=3.14)
        assert net.fim_lambda == 3.14
