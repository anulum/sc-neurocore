# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio network canvas nir

"""Focused suite: TestNIR from former test_studio_network_canvas.py."""

from __future__ import annotations

from tests.studio_network_canvas_support import *  # noqa: F403


class TestNIR:
    def test_export_nir(self):
        exc = create_population(label="E", count=80)
        inh = create_population(label="I", count=20, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        graph = {"populations": [exc, inh], "projections": [proj]}
        nir = graph_to_nir(graph)
        assert nir["format"] == "nir"
        assert len(nir["nodes"]) == 2
        assert len(nir["edges"]) == 1

    def test_import_nir(self):
        nir = {
            "format": "nir",
            "version": "0.1",
            "nodes": {
                "pop_a": {"type": "SCLapicqueLIFNeuron", "count": 80, "neuron_type": "excitatory"},
                "pop_b": {"type": "SCLapicqueLIFNeuron", "count": 20, "neuron_type": "inhibitory"},
            },
            "edges": [{"source": "pop_a", "target": "pop_b", "weight": 0.5}],
        }
        graph = nir_to_graph(nir)
        assert len(graph["populations"]) == 2
        assert len(graph["projections"]) == 1
        assert graph["populations"][0]["id"] == "pop_a"
        assert graph["populations"][0]["model"] == "SCLapicqueLIFNeuron"
        assert graph["projections"][0]["rule"] == "all_to_all"
        assert validate_graph(graph) == []

    def test_import_rejects_nir_primitive_types_and_inconsistent_graphs(self):
        with pytest.raises(ValueError, match="is not a catalogue model"):
            nir_to_graph({"nodes": {"a": {"type": "LIF", "count": 10}}, "edges": []})
        with pytest.raises(ValueError, match="not executable"):
            nir_to_graph(
                {
                    "nodes": {
                        "a": {
                            "type": "SCLapicqueLIFNeuron",
                            "count": 10,
                            "neuron_type": "inhibitory",
                        },
                        "b": {"type": "SCLapicqueLIFNeuron", "count": 10},
                    },
                    "edges": [{"source": "a", "target": "b", "weight": 1.0}],
                }
            )

    def test_roundtrip(self):
        exc = create_population(label="E", count=64)
        inh = create_population(label="I", count=16, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"], weight=0.3)
        graph = {"populations": [exc, inh], "projections": [proj]}
        nir = graph_to_nir(graph)
        assert nir["nodes"][exc["id"]]["type"] == "SCLapicqueLIFNeuron"
        restored = nir_to_graph(nir)
        assert len(restored["populations"]) == 2
        assert len(restored["projections"]) == 1
        # The format carries no probability: the restored edge is all-to-all.
        assert restored["projections"][0]["rule"] == "all_to_all"
