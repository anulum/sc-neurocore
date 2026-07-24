# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTopologicalAnalysis from former test_dna_mapper.py

"""Focused suite: TestTopologicalAnalysis from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestTopologicalAnalysis:
    """Circuit topology analysis."""

    def test_depth_single_gate(self, simple_and_circuit: DNACircuitDesign) -> None:
        analyzer = TopologicalAnalyzer()
        result = analyzer.analyze(simple_and_circuit)
        assert result["depth"] >= 1
        assert result["has_feedback"] is False

    def test_depth_cascade(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "NOT", "inputs": ["X"], "output": "Y"},
                {"type": "OR", "inputs": ["Y", "C"], "output": "Z"},
            ],
            input_names=["A", "B", "C"],
            output_names=["Z"],
        )
        result = TopologicalAnalyzer().analyze(design)
        assert result["depth"] >= 2
        assert result["n_nodes"] >= 4

    def test_fan_out_detected(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "NOT", "inputs": ["A"], "output": "Y"},
            ],
            input_names=["A", "B"],
            output_names=["X", "Y"],
        )
        result = TopologicalAnalyzer().analyze(design)
        assert result["fan_out"]["A"] >= 2

    def test_no_feedback_in_dag(self, simple_and_circuit: DNACircuitDesign) -> None:
        result = TopologicalAnalyzer().analyze(simple_and_circuit)
        assert result["has_feedback"] is False
        assert len(result["cycles"]) == 0

    def test_feedback_cycle_is_reported_from_remaining_nodes(self) -> None:
        design = DNACircuitDesign(
            name="cycle",
            gates=[
                DNAGate(0, GateType.BUFFER, ["A"], "B"),
                DNAGate(1, GateType.BUFFER, ["B"], "A"),
            ],
        )

        result = TopologicalAnalyzer().analyze(design)

        assert result["has_feedback"] is True
        assert result["cycles"] == [["A", "B"]]
