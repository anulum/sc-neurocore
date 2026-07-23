# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKineticSimulator from former test_bridges_dna_mapper.py

"""Focused suite: TestKineticSimulator from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403

class TestKineticSimulator:
    def test_simulate_produces_trajectory(self) -> None:
        compiler = StrandDisplacementCompiler()
        and_gate = compiler.compile_and("a", "b", "out")
        design = DNACircuitDesign(
            name="test_circuit",
            gates=[and_gate],
            input_strands=[
                DNAStrand(name="a", sequence="ACGT"),
                DNAStrand(name="b", sequence="TGCA"),
            ],
            output_strands=[DNAStrand(name="out", sequence="AGCT")],
        )
        sim = KineticSimulator()
        result = sim.simulate(
            design, input_concentrations={"a": 100.0, "b": 100.0}, duration_s=10.0, dt=1.0
        )
        assert isinstance(result, dict)
