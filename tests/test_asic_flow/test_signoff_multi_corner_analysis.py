# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-corner static timing script and slack tests

"""Verify per-PDK corner tables, real Liberty file names, and fail-closed refusals."""

from __future__ import annotations

import pytest

from sc_neurocore.asic_flow.asic_flow import (
    CornerType,
    DesignParams,
    MultiCornerAnalysis,
    PDKConfig,
    PDKType,
    PVTCorner,
)


def _liberty_basenames(script: str) -> list[str]:
    """Extract ordered Liberty basenames from read_liberty commands."""
    return [
        line.split()[1].rsplit("/", 1)[-1]
        for line in script.splitlines()
        if line.startswith("read_liberty ")
    ]


class TestMultiCornerAnalysis:
    """Validate per-PDK PVT corner Liberty tables and fail-closed boundaries."""

    def test_sky130_default_corners_match_real_library_names(self) -> None:
        """Sky130 default corners match skywater-pdk timing/*.lib names byte for byte."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        script = MultiCornerAnalysis.generate(pdk, DesignParams())

        assert _liberty_basenames(script) == [
            "sky130_fd_sc_hd__tt_025C_1v80.lib",
            "sky130_fd_sc_hd__ss_100C_1v60.lib",
            "sky130_fd_sc_hd__ss_n40C_1v60.lib",
            "sky130_fd_sc_hd__ff_n40C_1v95.lib",
            "sky130_fd_sc_hd__ff_100C_1v95.lib",
        ]

    def test_gf180mcu_default_corners_match_real_library_names(self) -> None:
        """GF180MCU 3.3 V default corners match gf180mcu_fd_sc_mcu7t5v0 real files."""
        pdk = PDKConfig.from_pdk_type(PDKType.GF180MCU)
        script = MultiCornerAnalysis.generate(pdk, DesignParams())

        assert _liberty_basenames(script) == [
            "gf180mcu_fd_sc_mcu7t5v0__tt_025C_3v30.lib",
            "gf180mcu_fd_sc_mcu7t5v0__ss_125C_3v00.lib",
            "gf180mcu_fd_sc_mcu7t5v0__ss_n40C_3v00.lib",
            "gf180mcu_fd_sc_mcu7t5v0__ff_n40C_3v60.lib",
            "gf180mcu_fd_sc_mcu7t5v0__ff_125C_3v60.lib",
        ]

    def test_pdks_without_corner_table_refuse_default_call(self) -> None:
        """TSMC28, Intel16, and custom PDKs refuse default corner generation."""
        for pdk_type in (PDKType.TSMC28, PDKType.INTEL16, PDKType.CUSTOM):
            pdk = PDKConfig.from_pdk_type(pdk_type)
            with pytest.raises(ValueError, match="corner table"):
                MultiCornerAnalysis.generate(pdk, DesignParams())

    def test_explicit_corners_check_real_table_for_open_pdks(self) -> None:
        """Open PDKs accept real table suffixes and reject foreign-family suffixes."""
        sky = PDKConfig.from_pdk_type(PDKType.SKY130)
        valid_sky = [PVTCorner(CornerType.TT, 100.0, 1.80, "_tt_100C_1v80")]
        assert _liberty_basenames(MultiCornerAnalysis.generate(sky, DesignParams(), valid_sky)) == [
            "sky130_fd_sc_hd__tt_100C_1v80.lib"
        ]

        with pytest.raises(ValueError, match="not in the verified"):
            MultiCornerAnalysis.generate(
                sky,
                DesignParams(),
                [PVTCorner(CornerType.SS, 125.0, 1.62, "_ss_125C_1v62")],
            )

        gf_3v3 = PDKConfig.from_pdk_type(PDKType.GF180MCU)
        with pytest.raises(ValueError, match="not in the verified"):
            MultiCornerAnalysis.generate(
                gf_3v3,
                DesignParams(),
                [PVTCorner(CornerType.SS, 125.0, 1.62, "_ss_125C_1v62")],
            )

        gf_1v8 = PDKConfig(
            pdk_type=PDKType.GF180MCU,
            liberty_file=(
                "$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/"
                "gf180mcu_fd_sc_mcu7t5v0__tt_025C_1v80.lib"
            ),
            voltage_v=1.8,
            nominal_liberty_suffix="_tt_025C_1v80",
        )
        assert _liberty_basenames(
            MultiCornerAnalysis.generate(
                gf_1v8,
                DesignParams(),
                [PVTCorner(CornerType.SS, 125.0, 1.62, "_ss_125C_1v62")],
            )
        ) == ["gf180mcu_fd_sc_mcu7t5v0__ss_125C_1v62.lib"]

    def test_custom_ihp_corners_and_closed_pdk_refusals(self) -> None:
        """Custom IHP substitutes _typ_* tokens; tokenless or mismatched paths fail."""
        ihp_corners = [
            PVTCorner(CornerType.SS, 125.0, 1.08, "_slow_1p08V_125C"),
            PVTCorner(CornerType.TT, 25.0, 1.20, "_typ_1p20V_25C"),
            PVTCorner(CornerType.FF, -40.0, 1.32, "_fast_1p32V_m40C"),
        ]
        for token in ("_typ_1p20V_25C", ""):
            ihp = PDKConfig(
                pdk_type=PDKType.CUSTOM,
                liberty_file="/opt/eda/orfs/ihp-sg13g2/lib/sg13g2_stdcell_typ_1p20V_25C.lib",
                nominal_liberty_suffix=token,
            )
            assert _liberty_basenames(
                MultiCornerAnalysis.generate(ihp, DesignParams(), ihp_corners)
            ) == [
                "sg13g2_stdcell_slow_1p08V_125C.lib",
                "sg13g2_stdcell_typ_1p20V_25C.lib",
                "sg13g2_stdcell_fast_1p32V_m40C.lib",
            ]

        corner = ihp_corners[0]
        for bad_pdk in (
            PDKConfig.from_pdk_type(PDKType.TSMC28),
            PDKConfig.from_pdk_type(PDKType.INTEL16),
            PDKConfig.from_pdk_type(PDKType.CUSTOM),
            PDKConfig(
                pdk_type=PDKType.CUSTOM,
                liberty_file="/opt/eda/orfs/ihp-sg13g2/lib/sg13g2_stdcell_typ_1p20V_25C.lib",
                nominal_liberty_suffix="_tt_025C_1v80",
            ),
        ):
            with pytest.raises(ValueError):
                MultiCornerAnalysis.generate(bad_pdk, DesignParams(), [corner])

    def test_worst_slack(self) -> None:
        """Smallest worst negative slack selects the limiting corner."""
        assert MultiCornerAnalysis.worst_slack({"tt": 0.5, "ss": -0.2, "ff": 1.0}) == ("ss", -0.2)

    def test_empty_slack_map_returns_neutral_sentinel(self) -> None:
        """No reported corners produce the documented neutral sentinel."""
        assert MultiCornerAnalysis.worst_slack({}) == ("none", 0.0)

    def test_corner_without_suffix_uses_configured_liberty(self) -> None:
        """An unsuffixed corner leaves the configured Liberty path unchanged."""
        pdk = PDKConfig(pdk_type=PDKType.CUSTOM, liberty_file="custom.lib")
        corner = PVTCorner(CornerType.TT, 25.0, 1.8)
        assert "read_liberty custom.lib" in MultiCornerAnalysis.generate(
            pdk, DesignParams(), [corner]
        )

    def test_gf180mcu_5v0_explicit_corners(self) -> None:
        """GF180MCU 5.0 V configurations accept verified 5.0 V-family suffixes."""
        gf_5v0 = PDKConfig(
            pdk_type=PDKType.GF180MCU,
            liberty_file=(
                "$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/"
                "gf180mcu_fd_sc_mcu7t5v0__tt_025C_5v00.lib"
            ),
            voltage_v=5.0,
            nominal_liberty_suffix="_tt_025C_5v00",
        )
        assert _liberty_basenames(
            MultiCornerAnalysis.generate(
                gf_5v0,
                DesignParams(),
                [PVTCorner(CornerType.SS, 125.0, 4.50, "_ss_125C_4v50")],
            )
        ) == ["gf180mcu_fd_sc_mcu7t5v0__ss_125C_4v50.lib"]

    def test_gf180mcu_unknown_nominal_suffix_raises(self) -> None:
        """GF180MCU refuses unknown or empty nominal suffixes without silent fallback."""
        corner = PVTCorner(CornerType.SS, 125.0, 3.00, "_ss_125C_3v00")
        for bad_nominal in ("_tt_025C_2v50", ""):
            bad_gf = PDKConfig(
                pdk_type=PDKType.GF180MCU,
                liberty_file=(
                    "$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/"
                    "gf180mcu_fd_sc_mcu7t5v0__tt_025C_3v30.lib"
                ),
                voltage_v=3.3,
                nominal_liberty_suffix=bad_nominal,
            )
            with pytest.raises(
                ValueError, match=f"unknown gf180mcu nominal_liberty_suffix {bad_nominal!r}"
            ):
                MultiCornerAnalysis.generate(bad_gf, DesignParams(), [corner])

    def test_pvt_corner_label(self) -> None:
        """Corner label combines process corner, temperature, and voltage."""
        assert PVTCorner(CornerType.SS, 125.0, 1.62).label == "ss_125C_1.62V"
