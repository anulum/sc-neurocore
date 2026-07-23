# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGateOptimizer from former test_dna_mapper.py

"""Focused suite: TestGateOptimizer from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestGateOptimizer:
    """Circuit-level gate optimization."""

    def test_no_change_normal_circuit(self) -> None:
        opt = GateOptimizer()
        gates = [
            {"type": "AND", "inputs": ["A", "B"], "output": "C"},
        ]
        result = opt.optimize(gates, ["C"])
        assert result["removed_count"] == 0
        assert len(result["optimized_gates"]) == 1

    def test_removes_duplicate(self) -> None:
        opt = GateOptimizer()
        gates = [
            {"type": "AND", "inputs": ["A", "B"], "output": "C"},
            {"type": "AND", "inputs": ["A", "B"], "output": "C"},
        ]
        result = opt.optimize(gates, ["C"])
        assert result["removed_count"] >= 1

    def test_identity_buffer_removal(self) -> None:
        opt = GateOptimizer()
        gates = [
            {"type": "AND", "inputs": ["A", "B"], "output": "C"},
            {"type": "BUFFER", "inputs": ["C"], "output": "D"},
            {"type": "NOT", "inputs": ["D"], "output": "dead"},
        ]
        result = opt.optimize(gates, ["C"])
        reasons = {removal["reason"] for removal in result["removals"]}
        assert {"identity_buffer", "dead_output"}.issubset(reasons)
