# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verified Liberty corner table and re-export tests

"""Verify sorted open-PDK Liberty suffix tables and signoff re-exports."""

from __future__ import annotations

from sc_neurocore.asic_flow import asic_flow, liberty_corners, pdk, signoff


class TestLibertyCorners:
    """Validate upstream Sky130 and GF180MCU Liberty tables and re-exports."""

    def test_sky130_liberty_suffixes_count_and_nominal_membership(self) -> None:
        """Sky130 table holds 16 base suffixes plus two ccsnoise variants."""
        assert isinstance(liberty_corners.SKY130_LIBERTY_SUFFIXES, frozenset)
        assert len(liberty_corners.SKY130_LIBERTY_SUFFIXES) == 18
        assert (
            liberty_corners.SKY130_NOMINAL_LIBERTY_SUFFIX in liberty_corners.SKY130_LIBERTY_SUFFIXES
        )
        assert {c.liberty_suffix for c in liberty_corners.DEFAULT_CORNERS} <= (
            liberty_corners.SKY130_LIBERTY_SUFFIXES
        )

    def test_gf180mcu_tables_cover_three_voltage_families(self) -> None:
        """GF180MCU maps 1.8 V, 3.3 V, and 5.0 V nominals to five corners each."""
        assert sorted(liberty_corners.GF180MCU_SUFFIXES_BY_NOMINAL) == [
            "_tt_025C_1v80",
            "_tt_025C_3v30",
            "_tt_025C_5v00",
        ]
        for nominal, table in liberty_corners.GF180MCU_SUFFIXES_BY_NOMINAL.items():
            assert len(table) == 5
            assert nominal in table
        assert {c.liberty_suffix for c in liberty_corners._GF180MCU_3V3_CORNERS} <= (
            liberty_corners.GF180MCU_SUFFIXES_BY_NOMINAL[
                liberty_corners.GF180MCU_3V3_NOMINAL_LIBERTY_SUFFIX
            ]
        )

    def test_signoff_and_pdk_reexports_preserve_object_identity(self) -> None:
        """Historical signoff, pdk, and facade imports resolve identical objects."""
        assert signoff.DEFAULT_CORNERS is liberty_corners.DEFAULT_CORNERS
        assert asic_flow.DEFAULT_CORNERS is liberty_corners.DEFAULT_CORNERS
        assert signoff.CornerType is liberty_corners.CornerType
        assert signoff.PVTCorner is liberty_corners.PVTCorner
        assert pdk.SKY130_LIBERTY_SUFFIXES is liberty_corners.SKY130_LIBERTY_SUFFIXES
        assert pdk.GF180MCU_SUFFIXES_BY_NOMINAL is liberty_corners.GF180MCU_SUFFIXES_BY_NOMINAL
