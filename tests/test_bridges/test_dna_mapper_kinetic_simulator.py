# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKineticSimulator from former test_dna_mapper.py

"""Focused suite: TestKineticSimulator from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestKineticSimulator:
    """Simulation correctness and convergence."""

    def test_and_both_inputs_high(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 200.0}, duration_s=3600.0)
        assert "time" in result
        output_key = simple_and_circuit.gates[0].output_name
        assert output_key in result
        final = result[output_key][-1]
        assert final > 50.0, f"AND(1,1) should produce high output, got {final}"

    def test_and_one_input_low(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 0.0}, duration_s=3600.0)
        output_key = simple_and_circuit.gates[0].output_name
        final = result[output_key][-1]
        assert final < 50.0, f"AND(1,0) should produce low output, got {final}"

    def test_simulation_time_steps(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 100.0}, duration_s=100.0, dt=0.5)
        assert len(result["time"]) == 200

    def test_concentrations_non_negative(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 200.0})
        for key, trace in result.items():
            if key == "time":
                continue
            assert np.all(trace >= 0), f"Negative concentrations in {key}"

    def test_concentrations_bounded(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 200.0})
        for key, trace in result.items():
            if key == "time":
                continue
            assert np.all(trace <= 201.0), f"Concentration exceeds max in {key}"
