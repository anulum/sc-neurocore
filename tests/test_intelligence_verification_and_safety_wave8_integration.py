# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWave8Integration from former test_intelligence_verification_and_safety.py

"""Focused suite: TestWave8Integration from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403

class TestWave8Integration:
    def test_nir_to_stability(self):
        from sc_neurocore.compiler.intelligence import (
            import_nir_graph,
            verify_ode_stability,
        )

        g = import_nir_graph(
            {
                "nodes": {"n0": {"type": "LIF", "tau": 10}},
                "edges": [],
            }
        )
        r = verify_ode_stability(g.equations, dt=0.1)
        assert r.stable is True

    def test_carbon_vs_reliability(self):
        from sc_neurocore.compiler.intelligence import (
            estimate_carbon_footprint,
            predict_reliability,
        )

        c = estimate_carbon_footprint("artix7", power_mw=500)
        r = predict_reliability(voltage_v=0.9, temperature_c=85)
        assert c.total_5yr_kg_co2 > 0
        assert r.mttf_years > 0

    def test_fault_tree_then_testbench(self):
        from sc_neurocore.compiler.intelligence import (
            generate_fault_tree,
            generate_testbench,
        )

        ft = generate_fault_tree("sc_lif", {"v": "a"})
        tb = generate_testbench("sc_lif", {"v": "a"})
        assert len(ft.mcs) > 0
        assert len(tb) > 100
