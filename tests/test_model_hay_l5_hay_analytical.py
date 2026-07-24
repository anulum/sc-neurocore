# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHayAnalytical from former test_model_hay_l5.py

"""Focused suite: TestHayAnalytical from former test_model_hay_l5.py."""

from __future__ import annotations

from tests.model_hay_l5_support import *  # noqa: F403


class TestHayAnalytical:
    def test_4_substeps(self) -> None:
        n = HayL5PyramidalNeuron()
        assert n.dt == 0.025  # 4 sub-steps in source

    def test_three_compartments(self) -> None:
        """Soma (v_s), trunk (v_t), tuft (v_a)."""
        n = HayL5PyramidalNeuron()
        assert hasattr(n, "v_s") and hasattr(n, "v_t") and hasattr(n, "v_a")

    def test_compartment_area_fractions(self) -> None:
        """p_s + p_t + p_a = 1.0 (area conservation)."""
        n = HayL5PyramidalNeuron()
        assert abs(n.p_s + n.p_t + n.p_a - 1.0) < 1e-12

    def test_coupling_soma_trunk(self) -> None:
        """g_st couples soma↔trunk bidirectionally."""
        n = HayL5PyramidalNeuron()
        assert n.g_st > 0

    def test_coupling_trunk_tuft(self) -> None:
        """g_ta couples trunk↔tuft bidirectionally."""
        n = HayL5PyramidalNeuron()
        assert n.g_ta > 0

    def test_ca_dynamics_in_tuft(self) -> None:
        """Ca dynamics: dCa = (-f_ca·I_Ca - Ca/ca_decay)·dt, clipped ≥ 0."""
        n = HayL5PyramidalNeuron()
        for _ in range(5000):
            n.step(10.0)
        assert n.ca_a >= 0

    def test_reversal_ordering(self) -> None:
        n = HayL5PyramidalNeuron()
        assert n.e_k < n.e_l < n.e_ih < n.e_na < n.e_ca

    def test_soma_currents(self) -> None:
        """Soma: Na, K, leak, coupling."""
        n = HayL5PyramidalNeuron()
        assert n.g_na > 0 and n.g_k > 0 and n.g_l_s > 0

    def test_trunk_currents(self) -> None:
        """Trunk: Ca, Ih, leak."""
        n = HayL5PyramidalNeuron()
        assert n.g_ca_t > 0 and n.g_ih > 0 and n.g_l_t > 0

    def test_tuft_currents(self) -> None:
        """Tuft: CaA, KCa, leak."""
        n = HayL5PyramidalNeuron()
        assert n.g_ca_a > 0 and n.g_kca > 0 and n.g_l_a > 0
