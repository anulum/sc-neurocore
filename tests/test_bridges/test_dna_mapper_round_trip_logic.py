# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRoundTripLogic from former test_dna_mapper.py

"""Focused suite: TestRoundTripLogic from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestRoundTripLogic:
    """Verify that compiled circuits implement correct Boolean logic."""

    def test_and_truth_table(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        sim = KineticSimulator()
        output_key = design.gates[0].output_name

        # (0, 0) → 0
        r = sim.simulate(design, {"A": 0.0, "B": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        # (1, 0) → 0
        r = sim.simulate(design, {"A": 200.0, "B": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        # (0, 1) → 0
        r = sim.simulate(design, {"A": 0.0, "B": 200.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        # (1, 1) → 1
        r = sim.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=1800.0)
        assert r[output_key][-1] > 50.0

    def test_or_truth_table(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "OR", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        sim = KineticSimulator()
        output_key = design.gates[0].output_name

        r = sim.simulate(design, {"A": 0.0, "B": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        r = sim.simulate(design, {"A": 200.0, "B": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] > 50.0

    def test_not_truth_table(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "NOT", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )
        sim = KineticSimulator()
        output_key = design.gates[0].output_name

        # Input high → output low
        r = sim.simulate(design, {"A": 200.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        # Input low → output high
        r = sim.simulate(design, {"A": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] > 50.0
