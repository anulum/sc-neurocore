# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SymbiYosys deployment contracts

"""Contracts for deployment-oriented SymbiYosys project generation."""

from __future__ import annotations


class TestSymbiYosys:
    """Tests for SymbiYosys .sby script generation."""

    def test_basic_bmc_script(self) -> None:
        """Default BMC script has required sections."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif")
        assert "[options]" in sby
        assert "mode bmc" in sby
        assert "depth 20" in sby
        assert "[engines]" in sby
        assert "smtbmc boolector" in sby
        assert "[script]" in sby
        assert "read_verilog -formal sc_lif.v" in sby
        assert "read_verilog -sv -formal sc_lif_sva.sv" in sby
        assert "prep -top sc_lif" in sby
        assert "[files]" in sby

    def test_prove_mode(self) -> None:
        """Prove mode sets induction."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif", mode="prove", depth=50)
        assert "mode prove" in sby
        assert "depth 50" in sby

    def test_cover_mode(self) -> None:
        """Cover mode for reachability."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif", mode="cover")
        assert "mode cover" in sby

    def test_custom_sva_file(self) -> None:
        """Custom SVA file path."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif", sva_file="my_props.sv")
        assert "my_props.sv" in sby

    def test_z3_engine(self) -> None:
        """Z3 solver engine."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif", engine="z3")
        assert "smtbmc z3" in sby
