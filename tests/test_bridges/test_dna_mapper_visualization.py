# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVisualization from former test_dna_mapper.py

"""Focused suite: TestVisualization from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestVisualization:
    """Circuit and kinetics visualization."""

    def test_circuit_diagram(self, simple_and_circuit: DNACircuitDesign) -> None:
        diagram = visualize_circuit(simple_and_circuit)
        assert "Circuit:" in diagram
        assert "INPUTS:" in diagram
        assert "OUTPUTS:" in diagram

    def test_circuit_diagram_renders_vertical_connector_for_multi_gate_cascade(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "BUFFER", "inputs": ["X"], "output": "Y"},
            ],
            input_names=["A", "B"],
            output_names=["Y"],
        )

        diagram = visualize_circuit(design)

        assert "\n    │\n" in diagram
        assert "AND" in diagram

    def test_kinetics_sparkline(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 200.0})
        chart = visualize_kinetics(result)
        assert "nM" in chart
        assert len(chart) > 0
