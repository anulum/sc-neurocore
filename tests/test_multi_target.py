# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for multi-target comparison

"""Tests for multi-target compilation comparison."""

from __future__ import annotations

from sc_neurocore.compiler.multi_target import (
    CompilationResult,
    compile_multi_target,
    format_comparison_table,
)


class TestMultiTarget:
    """Test multi-target compilation and table formatting."""

    def test_compile_multi_target(self) -> None:
        """Should return results for multiple targets."""
        eqs = {"v": "a * b + c"}
        targets = ["artix7", "loihi2"]
        results = compile_multi_target(eqs, targets)
        assert len(results) == 2
        assert results[0].target == "artix7"
        assert results[1].target == "loihi2"
        assert results[0].data_width == 18
        assert results[1].data_width == 24

    def test_table_formatting(self) -> None:
        """Should produce a valid markdown table."""
        results = [
            CompilationResult(
                target="target1",
                verilog_lines=100,
                data_width=16,
                fraction=8,
                overflow="wrap",
                rounding="floor",
                estimated_luts=50,
                estimated_dsps=2,
                estimated_ffs=32,
                guard_bits=2,
                max_freq_mhz=200,
            )
        ]
        table = format_comparison_table(results)
        assert "| Target |" in table
        assert "target1" in table
        assert "200" in table

    def test_guard_bits_reflected(self) -> None:
        """Guard bits from static analysis should be present."""
        eqs = {"v": "a + b + c + d + e"}
        results = compile_multi_target(eqs, ["artix7"])
        assert results[0].guard_bits >= 2
