# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVerilogHeader from former test_gen_vmin_lif_lut.py

"""Focused suite: TestVerilogHeader from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

from tools.gen_vmin_lif_lut import (
    emit_lut_verilog_header,
    gen_softplus_lut,
    LUT_RANGE,
    LUT_SIZE,
)


class TestVerilogHeader:
    """Render the softplus table as a Verilog header."""

    def test_header_contains_size_define(self) -> None:
        """The header defines the table size."""
        lut = gen_softplus_lut(1.0, LUT_SIZE, LUT_RANGE)
        header = emit_lut_verilog_header(lut)
        assert f"`define VMIN_LUT_SIZE {LUT_SIZE}" in header

    def test_header_contains_all_entries(self) -> None:
        """The header defines every table entry."""
        lut = gen_softplus_lut(1.0, LUT_SIZE, LUT_RANGE)
        header = emit_lut_verilog_header(lut)
        for i in range(LUT_SIZE):
            assert f"`define VMIN_LUT_{i:02d}" in header

    def test_header_uses_signed_q88_literals(self) -> None:
        """Entries are written as signed 16-bit literals."""
        lut = gen_softplus_lut(1.0, LUT_SIZE, LUT_RANGE)
        header = emit_lut_verilog_header(lut)
        assert "16'sd" in header

    def test_header_has_provenance_comment(self) -> None:
        """The header carries the licence lines and a generated-file warning."""
        lut = gen_softplus_lut(1.0, LUT_SIZE, LUT_RANGE)
        header = emit_lut_verilog_header(lut)
        assert "// SPDX-License-Identifier: AGPL-3.0-or-later\n" in header
        assert "// Commercial license available\n" in header
        assert "SPDX-License-Identifier: AGPL-3.0-or-later |" not in header
        assert "Auto-generated" in header
        assert "DO NOT EDIT" in header
