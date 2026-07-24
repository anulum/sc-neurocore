# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNewGateTypes from former test_dna_mapper.py

"""Focused suite: TestNewGateTypes from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestNewGateTypes:
    """MUX, AMPLIFIER, and BUFFER gate compilation and simulation."""

    def test_mux_compiles(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "MUX", "inputs": ["S", "A", "B"], "output": "Y"}],
            input_names=["S", "A", "B"],
            output_names=["Y"],
        )
        assert design.total_gates == 1
        assert design.gates[0].gate_type == GateType.MUX
        assert len(design.gates[0].input_names) == 3

    def test_amplifier_compiles(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AMPLIFIER", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.total_gates == 1
        assert design.gates[0].gate_type == GateType.AMPLIFIER

    def test_buffer_compiles(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "BUFFER", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.total_gates == 1
        assert design.gates[0].gate_type == GateType.BUFFER

    def test_mux_strand_count(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_mux("S", "A", "B", "Y")
        assert gate.strand_count >= 4

    def test_amplifier_high_fuel(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_amplifier("A", "B")
        fuel = [s for s in gate.strands if s.role == "fuel"]
        assert len(fuel) >= 1
        assert fuel[0].concentration_nM >= 500.0

    def test_buffer_in_cascade(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "BUFFER", "inputs": ["X"], "output": "Y"},
                {"type": "NOT", "inputs": ["Y"], "output": "Z"},
            ],
            input_names=["A", "B"],
            output_names=["Z"],
        )
        assert design.total_gates == 3

    @pytest.mark.parametrize(
        ("gate_type", "inputs", "concentrations"),
        [
            (GateType.THRESHOLD, ["A"], {"A": 200.0}),
            (GateType.MUX, ["S", "A", "B"], {"S": 200.0, "A": 200.0, "B": 0.0}),
            (GateType.AMPLIFIER, ["A"], {"A": 50.0}),
            (GateType.BUFFER, ["A"], {"A": 200.0}),
        ],
    )
    def test_kinetic_simulator_dispatches_extended_gate_rates(
        self,
        gate_type: GateType,
        inputs: list[str],
        concentrations: dict[str, float],
    ) -> None:
        gate = DNAGate(
            gate_id=0,
            gate_type=gate_type,
            input_names=inputs,
            output_name="Y",
            threshold=0.5,
            leak_rate=0.0,
        )
        design = DNACircuitDesign(name="extended_gate", gates=[gate])

        result = KineticSimulator().simulate(design, concentrations, duration_s=20.0, dt=1.0)

        assert result["Y"][-1] > 0.0

    def test_kinetic_simulator_unknown_gate_fails_closed_to_leak_only(self) -> None:
        gate = DNAGate(
            gate_id=0,
            gate_type=GateType.CATALYTIC,
            input_names=["A"],
            output_name="Y",
            leak_rate=0.0,
        )
        design = DNACircuitDesign(name="unknown_gate", gates=[gate])

        result = KineticSimulator().simulate(design, {"A": 200.0}, duration_s=20.0, dt=1.0)

        assert np.all(result["Y"] == 0.0)

    def test_enzymatic_xor_network_compiles(self) -> None:
        c = BitstreamToDNA(method="enzymatic", seed=42)
        design = c.compile_network(
            gates=[{"type": "XOR", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )

        assert design.gates[0].gate_type == GateType.XOR
