# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWeightFormatConsistency from former test_e2e_pipeline.py

"""Focused suite: TestWeightFormatConsistency from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403


@pytest.mark.e2e
class TestWeightFormatConsistency:
    """All weight formats contain identical data."""

    def test_all_formats_same_values(self):
        """Verilog, .coe, .mif all encode the same weights."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        weights = [[100, 50, 0], [25, 75, 127]]

        v = generate_weight_rom(weights, data_width=16, output_format="verilog")
        coe = generate_weight_rom(weights, data_width=16, output_format="coe")
        mif = generate_weight_rom(weights, data_width=16, output_format="mif")

        # Extract hex values from each format
        hex_v = re.findall(r"'sh([0-9a-fA-F]+)", v)
        hex_coe = re.findall(r"^([0-9a-fA-F]+)[,;]", coe, re.MULTILINE)
        hex_mif = re.findall(r": ([0-9a-fA-F]+);", mif)

        assert len(hex_v) == 6
        assert len(hex_coe) == 6
        assert len(hex_mif) == 6
        assert hex_v == hex_coe == hex_mif
