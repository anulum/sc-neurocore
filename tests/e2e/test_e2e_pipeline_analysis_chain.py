# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAnalysisChain from former test_e2e_pipeline.py

"""Focused suite: TestAnalysisChain from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403


@pytest.mark.e2e
class TestAnalysisChain:
    """Pipeline depth → power → thermal → multi-target: all consistent."""

    def test_complex_ode_analysis_chain(self):
        """HH-class ODE: pipeline → power → thermal → compare."""
        from sc_neurocore.compiler.static_analysis import (
            critical_path_depth,
            pipeline_stages_needed,
            compute_guard_bits,
        )
        from sc_neurocore.compiler.deployment import compile_multi_target

        hh_expr = "gNa * m * m * m * h * (v - ENa)"
        depth = critical_path_depth(hh_expr)
        assert depth >= 3

        stages = pipeline_stages_needed(depth, 900)
        assert stages >= 1

        guard = compute_guard_bits(hh_expr)
        assert guard >= 0

        results = compile_multi_target(
            {"v": hh_expr},
            ["artix7", "ecp5"],
        )
        assert len(results) == 2
        assert len({r.guard_bits for r in results}) == 1

    def test_slr_placement_valid(self):
        """SLR placement constraints are structurally valid."""
        from sc_neurocore.compiler.deployment import (
            generate_slr_constraints,
            SLRPlacement,
        )

        placements = [
            SLRPlacement(module_name="sc_hh_v", slr=0, pblock_name="pblock_v"),
            SLRPlacement(module_name="sc_hh_n", slr=1, pblock_name="pblock_n"),
        ]
        xdc = generate_slr_constraints(placements)
        assert "pblock" in xdc.lower()
        assert "SLR0" in xdc
        assert "SLR1" in xdc
