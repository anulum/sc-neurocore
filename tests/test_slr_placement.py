# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SLR placement constraints

"""Tests for SLR placement constraint generation."""

from __future__ import annotations

from sc_neurocore.compiler.slr_placement import (
    SLRPlacement,
    generate_slr_constraints,
)


class TestSlrPlacement:
    """Test SLR placement constraint generation."""

    def test_single_slr(self) -> None:
        """Should produce XDC for single SLR placement."""
        placements = [SLRPlacement("sc_lif_0", 0)]
        xdc = generate_slr_constraints(placements)
        assert "create_pblock pblock_slr0" in xdc
        assert "resize_pblock [get_pblocks pblock_slr0] -add SLR0" in xdc
        assert "sc_lif_0" in xdc

    def test_multi_slr(self) -> None:
        """Should produce XDC for multiple SLR placements."""
        placements = [
            SLRPlacement("sc_lif_0", 0),
            SLRPlacement("sc_lif_1", 1),
        ]
        xdc = generate_slr_constraints(placements)
        assert "pblock_slr0" in xdc
        assert "pblock_slr1" in xdc
        assert "SLR0" in xdc
        assert "SLR1" in xdc

    def test_pipeline_regs_inserted(self) -> None:
        """Should insert pipeline register duplication for multi-SLR."""
        placements = [
            SLRPlacement("sc_lif_0", 0),
            SLRPlacement("sc_lif_1", 1),
        ]
        xdc = generate_slr_constraints(placements, insert_pipeline_regs=True)
        assert "REGISTER_DUPLICATION" in xdc
        assert "set_max_delay" in xdc

    def test_custom_pblock_name(self) -> None:
        """Should use custom pblock name if provided."""
        placements = [SLRPlacement("sc_lif_0", 0, pblock_name="CORE_PBLOCK")]
        xdc = generate_slr_constraints(placements)
        assert "CORE_PBLOCK" in xdc
