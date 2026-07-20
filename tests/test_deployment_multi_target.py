# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-target deployment contracts

"""Contracts for multi-target deployment comparison."""

from __future__ import annotations


class TestMultiTarget:
    """Tests for multi-target --compare compilation."""

    def test_basic_multi_target(self) -> None:
        """Compile LIF to 3 targets and get results."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "-(v - v_rest) / tau + R * I"},
            ["artix7", "loihi2", "asic_16"],
        )
        assert len(results) == 3
        targets = [r.target for r in results]
        assert "artix7" in targets
        assert "loihi2" in targets
        assert "asic_16" in targets

    def test_data_widths_differ(self) -> None:
        """Different targets have different data widths."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a * b + c"},
            ["artix7", "loihi2"],
        )
        r_map = {r.target: r for r in results}
        assert r_map["artix7"].data_width != r_map["loihi2"].data_width

    def test_guard_bits_consistent(self) -> None:
        """Guard bits should be same for all targets (expression-dependent)."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a + b + c + d"},
            ["artix7", "ice40", "ecp5"],
        )
        guards = [r.guard_bits for r in results]
        assert len(set(guards)) == 1  # All same

    def test_format_comparison_table(self) -> None:
        """Table formatter produces markdown."""
        from sc_neurocore.compiler.deployment import (
            compile_multi_target,
            format_comparison_table,
        )

        results = compile_multi_target(
            {"v": "a * b + c"},
            ["artix7", "ice40"],
        )
        table = format_comparison_table(results)
        assert "| Target" in table
        assert "artix7" in table
        assert "ice40" in table

    def test_single_target(self) -> None:
        """Single target still works."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a + b"},
            ["artix7"],
        )
        assert len(results) == 1
        assert results[0].target == "artix7"

    def test_dsp_allocation(self) -> None:
        """Targets with DSP blocks allocate multipliers to DSPs."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a * b * c"},
            ["artix7"],  # has DSP48E1
        )
        assert results[0].estimated_dsps > 0
