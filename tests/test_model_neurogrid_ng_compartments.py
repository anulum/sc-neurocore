# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNGCompartments from former test_model_neurogrid.py

"""Focused suite: TestNGCompartments from former test_model_neurogrid.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.neurogrid import (
    NeuroGridNeuron,
)


class TestNGCompartments:
    """The soma and dendrite compartments of the NeuroGridNeuron and their coupling."""

    def test_dendrite_slower_than_soma(self) -> None:
        """tau_d > tau_s → dendrite integrates slower."""
        n = NeuroGridNeuron()
        assert n.tau_d > n.tau_s

    def test_dendrite_accumulates(self) -> None:
        """Dendrite accumulates."""
        n = NeuroGridNeuron()
        vd_vals = []
        for _ in range(500):
            n.step(50.0)
            vd_vals.append(n.v_d)
        # Should depolarise from -65 toward steady state
        assert vd_vals[-1] > vd_vals[0]

    def test_coupling_transfers_charge(self) -> None:
        """With g_c=0, compartments are independent."""
        n = NeuroGridNeuron(g_c=0.0)
        for _ in range(1000):
            n.step(50.0)
        # Dendrite gets input, soma gets nothing (only exp term)
        assert n.v_d > n.v_rest
