# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiTargetConsistency from former test_e2e_pipeline.py

"""Focused suite: TestMultiTargetConsistency from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403


@pytest.mark.e2e
class TestMultiTargetConsistency:
    """Cross-target comparison: same ODE, different targets."""

    def test_guard_bits_target_independent(self):
        """Guard bits depend on the expression, not the target."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a + b + c + d"},
            ["artix7", "loihi2", "ecp5", "asic_16"],
        )
        guards = {r.guard_bits for r in results}
        assert len(guards) == 1

    def test_dsps_scale_with_dsp_block(self):
        """Targets with DSP blocks allocate DSPs."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a * b * c"},
            ["artix7", "ecp5"],
        )
        # Both have DSP blocks
        for r in results:
            assert r.estimated_dsps > 0

    def test_table_includes_all_targets(self):
        """Comparison table mentions every target."""
        from sc_neurocore.compiler.deployment import (
            compile_multi_target,
            format_comparison_table,
        )

        targets = ["artix7", "loihi2", "ecp5", "asic_16"]
        results = compile_multi_target({"v": "a * b + c"}, targets)
        table = format_comparison_table(results)
        for t in targets:
            assert t in table
