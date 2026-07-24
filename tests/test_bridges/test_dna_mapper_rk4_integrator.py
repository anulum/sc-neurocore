# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRK4Integrator from former test_dna_mapper.py

"""Focused suite: TestRK4Integrator from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestRK4Integrator:
    """RK4 vs Euler integrator comparison."""

    def test_rk4_produces_output(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        sim = KineticSimulator(integrator="rk4")
        result = sim.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=1800.0)
        assert result["C"][-1] > 50.0

    def test_rk4_matches_euler_qualitatively(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        euler = KineticSimulator(integrator="euler")
        rk4 = KineticSimulator(integrator="rk4")
        r_euler = euler.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=1800.0)
        r_rk4 = rk4.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=1800.0)
        assert abs(r_euler["C"][-1] - r_rk4["C"][-1]) < 30.0

    def test_temperature_affects_rate(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        sim_37 = KineticSimulator(temperature_c=37.0)
        sim_25 = KineticSimulator(temperature_c=25.0)
        r_37 = sim_37.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=600.0)
        r_25 = sim_25.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=600.0)
        # Higher temperature → faster kinetics
        assert r_37["C"][-1] > r_25["C"][-1]
