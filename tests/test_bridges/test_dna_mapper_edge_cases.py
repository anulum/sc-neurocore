# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeCases from former test_dna_mapper.py

"""Focused suite: TestEdgeCases from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_single_not_gate(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "NOT", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.total_gates == 1

    def test_deep_cascade_10_gates(self) -> None:
        c = BitstreamToDNA(seed=42)
        gates = []
        prev = "A"
        for i in range(10):
            out = f"g{i}"
            gates.append({"type": "NOT", "inputs": [prev], "output": out})
            prev = out
        design = c.compile_network(gates=gates, input_names=["A"], output_names=[prev])
        assert design.total_gates == 10

    def test_threshold_zero(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "THRESHOLD", "inputs": ["A"], "output": "B", "threshold": 0.0}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.gates[0].threshold == 0.0

    def test_threshold_one(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "THRESHOLD", "inputs": ["A"], "output": "B", "threshold": 1.0}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.gates[0].threshold == 1.0

    def test_high_level_simulate_wrapper_returns_time_and_gate_trace(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "BUFFER", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )

        result = c.simulate(design, {"A": 200.0}, duration_s=10.0, dt=1.0)

        assert "time" in result
        assert "B" in result
        assert result["B"][-1] > 0.0
