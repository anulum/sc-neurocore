# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SymbiYosys formal flow

"""Tests for SymbiYosys formal verification script generation."""

from __future__ import annotations

from sc_neurocore.compiler.sby_formal import generate_sby_script


class TestSbyFormal:
    """Test SymbiYosys script generation."""

    def test_basic_sby(self) -> None:
        """Should produce a valid .sby script."""
        sby = generate_sby_script("sc_lif")
        assert "[options]" in sby
        assert "mode bmc" in sby
        assert "depth 20" in sby
        assert "sc_lif.v" in sby

    def test_custom_mode_depth(self) -> None:
        """Custom mode and depth should propagate."""
        sby = generate_sby_script("sc_lif", mode="prove", depth=50)
        assert "mode prove" in sby
        assert "depth 50" in sby

    def test_cover_mode(self) -> None:
        """Cover mode should be supported."""
        sby = generate_sby_script("sc_lif", mode="cover")
        assert "mode cover" in sby

    def test_custom_sva_file(self) -> None:
        """Custom SVA file name should propagate."""
        sby = generate_sby_script("sc_lif", sva_file="my_props.sv")
        assert "my_props.sv" in sby

    def test_custom_engine(self) -> None:
        """Custom solver engine should propagate."""
        sby = generate_sby_script("sc_lif", engine="z3")
        assert "smtbmc z3" in sby
