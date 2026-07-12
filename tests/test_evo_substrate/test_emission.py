# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary artefact emission tests

"""Evolutionary artefact emission tests."""

from __future__ import annotations

from sc_neurocore.evo_substrate.emission import OrganismEmitter
from sc_neurocore.evo_substrate.genome import Genome


class TestOrganismEmitter:
    def test_to_nir(self) -> None:
        g = Genome()
        g.compute_id()
        nir = OrganismEmitter.to_nir(g)
        assert "nodes" in nir
        assert "edges" in nir
        assert len(nir["nodes"]) == g.topology.num_neurons

    def test_nir_has_arcane_params(self) -> None:
        g = Genome()
        g.compute_id()
        nir = OrganismEmitter.to_nir(g)
        node = list(nir["nodes"].values())[0]
        assert node["type"] == "ArcaneNeuron"
        assert "tau_fast" in node

    def test_to_verilog(self) -> None:
        g = Genome()
        g.compute_id()
        v = OrganismEmitter.to_verilog(g)
        assert "module" in v
        assert f"NUM_NEURONS = {g.topology.num_neurons}" in v
        assert "sc_lif_neuron" in v

    def test_verilog_custom_name(self) -> None:
        g = Genome()
        g.compute_id()
        v = OrganismEmitter.to_verilog(g, module_name="test_org")
        assert "module test_org" in v

    def test_to_photonic_netlist_carries_geometry_and_waveguides(self) -> None:
        g = Genome()
        g.topology.num_neurons = 3
        g.compute_id()

        netlist = OrganismEmitter.to_photonic_netlist(g, pml_layers=16)

        assert netlist["metadata"]["genome_id"] == g.genome_id
        assert netlist["parameters"]["pml_layers"] == 16
        assert [waveguide["id"] for waveguide in netlist["waveguides"]] == [
            "wg_0",
            "wg_1",
            "wg_2",
        ]


# ── Crossover Tests ─────────────────────────────────────────────────
