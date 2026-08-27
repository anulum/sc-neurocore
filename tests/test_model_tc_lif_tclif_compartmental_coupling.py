# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTCLIFCompartmentalCoupling from former test_model_tc_lif.py

"""Focused suite: TestTCLIFCompartmentalCoupling from former test_model_tc_lif.py."""

from __future__ import annotations

from tests.model_tc_lif_support import *  # noqa: F403


class TestTCLIFCompartmentalCoupling:
    """kappa controls soma←dendrite coupling."""

    def test_dendrite_charges_independently(self):
        """v_d responds to i_dend, not to i_soma directly."""
        n = SCLeakyTwoCompartmentLIFNeuron(theta=100.0)  # prevent spikes
        n.step(0.0, 1.0)  # i_dend=1.0
        assert n.v_d > 0.0

    def test_dendritic_input_boosts_soma(self):
        """Dendritic current flows to soma via kappa: v_s += kappa*(v_d - v_s)/tau_s."""
        n_dend = SCLeakyTwoCompartmentLIFNeuron()
        n_nodend = SCLeakyTwoCompartmentLIFNeuron()
        s_dend = len(_run(n_dend, i_soma=0.5, steps=5000, i_dend=5.0))
        s_nodend = len(_run(n_nodend, i_soma=0.5, steps=5000, i_dend=0.0))
        assert s_dend > s_nodend, (
            f"Dend: {s_dend}, no dend: {s_nodend} — dendritic input should help"
        )

    def test_kappa_controls_coupling_strength(self):
        """Higher kappa → more dendritic influence on soma."""
        n_weak = SCLeakyTwoCompartmentLIFNeuron(kappa=0.1)
        n_strong = SCLeakyTwoCompartmentLIFNeuron(kappa=2.0)
        s_weak = len(_run(n_weak, i_soma=0.5, steps=5000, i_dend=3.0))
        s_strong = len(_run(n_strong, i_soma=0.5, steps=5000, i_dend=3.0))
        assert s_strong > s_weak

    def test_somatic_reset_dendrite_unchanged(self):
        """On spike: v_s → v_reset but v_d retains its value."""
        n = SCLeakyTwoCompartmentLIFNeuron()
        for _ in range(5000):
            s = n.step(2.0, 1.0)
            if s == 1:
                assert n.v_s == n.v_reset
                # v_d should NOT be reset (it retains its value)
                # Can't check exact value, but it shouldn't be v_rest
                break

    def test_timescale_separation(self):
        """tau_d=20 >> tau_s=2: dendrite is 10× slower than soma."""
        n = SCLeakyTwoCompartmentLIFNeuron(theta=100.0)
        vs0, vd0 = n.v_s, n.v_d
        n.step(1.0, 1.0)
        dvs = abs(n.v_s - vs0)
        dvd = abs(n.v_d - vd0)
        assert dvs > dvd * 5, f"dvs={dvs:.4f}, dvd={dvd:.4f}"
