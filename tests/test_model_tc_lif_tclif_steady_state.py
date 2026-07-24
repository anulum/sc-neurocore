# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTCLIFSteadyState from former test_model_tc_lif.py

"""Focused suite: TestTCLIFSteadyState from former test_model_tc_lif.py."""

from __future__ import annotations

from tests.model_tc_lif_support import *  # noqa: F403


class TestTCLIFSteadyState:
    def test_soma_steady_state(self):
        """At steady state (no spikes): v_s_ss depends on both i_soma and v_d."""
        n = TwoCompartmentLIFNeuron(theta=100.0)  # prevent spikes
        for _ in range(10000):
            n.step(0.5, 0.0)
        # Soma steady state with v_d=0: v_s_ss = i_soma / (1 + kappa)
        # From: 0 = (-(v_s - 0) + kappa*(0 - v_s) + I)/tau_s
        # → 0 = -v_s(1+kappa) + I → v_s = I/(1+kappa) = 0.5/1.5 ≈ 0.333
        v_ss = 0.5 / (1.0 + n.kappa)
        assert abs(n.v_s - v_ss) < 0.01, f"v_s={n.v_s:.4f}, expected={v_ss:.4f}"

    def test_dendrite_steady_state(self):
        """v_d_ss = i_dend (at v_rest=0, from -(v_d-0)+i_dend=0 → v_d=i_dend)."""
        n = TwoCompartmentLIFNeuron(theta=100.0)
        for _ in range(10000):
            n.step(0.0, 2.0)
        # v_d_ss = i_dend = 2.0
        assert abs(n.v_d - 2.0) < 0.01
