# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SLR placement deployment contracts

"""Contracts for deployment SLR placement and constraints."""

from __future__ import annotations


class TestSLRPlacement:
    """Tests for multi-die SLR constraint generation."""

    def test_single_slr(self) -> None:
        """Single SLR placement generates PBLOCK."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement,
            generate_slr_constraints,
        )

        xdc = generate_slr_constraints(
            [
                SLRPlacement("neuron_array", slr=0),
            ]
        )
        assert "create_pblock pblock_slr0" in xdc
        assert "SLR0" in xdc
        # No inter-SLR directives for single SLR
        assert "REGISTER_DUPLICATION" not in xdc

    def test_multi_slr_pipeline_regs(self) -> None:
        """Multi-SLR adds pipeline register directives."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement,
            generate_slr_constraints,
        )

        xdc = generate_slr_constraints(
            [
                SLRPlacement("input_stage", slr=0),
                SLRPlacement("compute_stage", slr=1),
            ]
        )
        assert "SLR0" in xdc
        assert "SLR1" in xdc
        assert "REGISTER_DUPLICATION" in xdc
        assert "set_max_delay" in xdc

    def test_no_pipeline_regs(self) -> None:
        """Opt-out of pipeline register insertion."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement,
            generate_slr_constraints,
        )

        xdc = generate_slr_constraints(
            [SLRPlacement("a", 0), SLRPlacement("b", 1)],
            insert_pipeline_regs=False,
        )
        assert "REGISTER_DUPLICATION" not in xdc

    def test_custom_pblock_name(self) -> None:
        """Custom PBLOCK name."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement,
            generate_slr_constraints,
        )

        xdc = generate_slr_constraints(
            [
                SLRPlacement("core", slr=2, pblock_name="pblock_core"),
            ]
        )
        assert "create_pblock pblock_core" in xdc

    def test_auto_pblock_name(self) -> None:
        """Auto-generated PBLOCK name from SLR index."""
        from sc_neurocore.compiler.deployment import SLRPlacement

        p = SLRPlacement("test", slr=3)
        assert p.pblock_name == "pblock_slr3"
