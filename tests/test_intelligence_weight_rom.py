# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight ROM generation contracts

"""Contracts for generated compiler weight ROMs."""

from __future__ import annotations

import pytest


class TestWeightROM:
    """Tests for synaptic weight ROM generation."""

    def test_verilog_rom(self) -> None:
        """Verilog ROM module."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[1, 2], [3, 4]]
        v = generate_weight_rom(w)
        assert "module sc_weight_rom" in v
        assert "case" in v
        assert "endmodule" in v

    def test_coe_format(self) -> None:
        """Xilinx .coe format."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[10, 20], [30, 40]]
        coe = generate_weight_rom(w, output_format="coe")
        assert "memory_initialization_radix=16" in coe
        assert "memory_initialization_vector=" in coe

    def test_mif_format(self) -> None:
        """Intel .mif format."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[10, 20], [30, 40]]
        mif = generate_weight_rom(w, output_format="mif")
        assert "WIDTH=16" in mif
        assert "DEPTH=4" in mif
        assert "CONTENT BEGIN" in mif
        assert "END;" in mif

    def test_custom_module_name(self) -> None:
        """Custom ROM module name."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[1, 2]]
        v = generate_weight_rom(w, module_name="my_weights")
        assert "module my_weights" in v

    def test_correct_entry_count(self) -> None:
        """Correct number of entries in ROM."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[1, 2, 3], [4, 5, 6]]
        mif = generate_weight_rom(w, output_format="mif")
        assert "DEPTH=6" in mif

    def test_data_width_propagates(self) -> None:
        """Custom data width propagates."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[1]]
        mif = generate_weight_rom(w, data_width=8, output_format="mif")
        assert "WIDTH=8" in mif

    def test_rejects_unknown_output_format(self) -> None:
        """Unknown memory formats must not silently emit Verilog."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        with pytest.raises(ValueError, match="Unsupported weight ROM format"):
            generate_weight_rom([[1]], output_format="hex")
