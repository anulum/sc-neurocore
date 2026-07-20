# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler NIR-import contracts

"""Contracts for importing NIR graphs into compiler representations."""

from __future__ import annotations


class TestNIRImport:
    def test_lif_import(self) -> None:
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph(
            {
                "nodes": {"n0": {"type": "LIF", "tau": 20}},
                "edges": [],
            }
        )
        assert "n0" in g.equations
        assert "20" in g.equations["n0"]

    def test_edges(self) -> None:
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph(
            {
                "nodes": {"a": {"type": "LIF"}, "b": {"type": "LIF"}},
                "edges": [["a", "b"]],
            }
        )
        assert ("a", "b") in g.edges

    def test_framework(self) -> None:
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph({"nodes": {}, "edges": []}, framework="Norse")
        assert g.framework == "Norse"

    def test_izhikevich_import(self) -> None:
        """An Izhikevich node maps to its quadratic membrane equation."""
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph({"nodes": {"n0": {"type": "Izhikevich"}}, "edges": []})
        assert "0.04" in g.equations["n0"]

    def test_unknown_type_falls_back_to_leaky_equation(self) -> None:
        """An unrecognised node type falls back to a generic leaky equation."""
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph({"nodes": {"n0": {"type": "Mystery", "tau": 5.0}}, "edges": []})
        assert "5.0" in g.equations["n0"]
