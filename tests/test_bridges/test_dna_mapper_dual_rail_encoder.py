# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDualRailEncoder from former test_dna_mapper.py

"""Focused suite: TestDualRailEncoder from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestDualRailEncoder:
    """Dual-rail fault-tolerant encoding."""

    def test_doubles_gate_count(self, simple_and_circuit: DNACircuitDesign) -> None:
        encoder = DualRailEncoder()
        compiler = BitstreamToDNA(seed=42)
        dual = encoder.encode(simple_and_circuit, compiler)
        assert dual.total_gates == simple_and_circuit.total_gates * 2

    def test_dual_rail_name(self, simple_and_circuit: DNACircuitDesign) -> None:
        encoder = DualRailEncoder()
        compiler = BitstreamToDNA(seed=42)
        dual = encoder.encode(simple_and_circuit, compiler)
        assert "dual_rail" in dual.name

    def test_fault_detection_no_faults(self) -> None:
        encoder = DualRailEncoder()
        result = {
            "time": np.array([0, 1]),
            "X_T": np.array([200.0, 200.0]),
            "X_C": np.array([0.0, 0.0]),
        }
        faults = encoder.check_faults(result)
        assert len(faults) == 0

    def test_fault_detection_stuck_high(self) -> None:
        encoder = DualRailEncoder()
        result = {
            "time": np.array([0, 1]),
            "X_T": np.array([200.0, 200.0]),
            "X_C": np.array([200.0, 200.0]),
        }
        faults = encoder.check_faults(result)
        assert len(faults) == 1
        assert faults[0]["fault_type"] == "stuck_high"

    def test_fault_detection_ignores_incomplete_dual_rail_pair(self) -> None:
        encoder = DualRailEncoder()
        result = {
            "time": np.array([0, 1]),
            "X_T": np.array([0.0, 200.0]),
        }

        assert encoder.check_faults(result) == []
