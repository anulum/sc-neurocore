# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verified Liberty corner tables for open-source PDKs

"""Define verified Liberty corner tables and PVT corner descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class CornerType(Enum):
    """Process-corner combinations used by multi-corner timing analysis."""

    TT = "tt"  # typical
    FF = "ff"  # fast-fast
    SS = "ss"  # slow-slow
    SF = "sf"  # slow-fast
    FS = "fs"


@dataclass
class PVTCorner:
    """Process-Voltage-Temperature corner definition."""

    corner: CornerType
    temperature_c: float
    voltage_v: float
    liberty_suffix: str = ""
    is_signoff: bool = True

    @property
    def label(self) -> str:
        """Return a stable corner-temperature-voltage label."""
        return f"{self.corner.value}_{self.temperature_c:.0f}C_{self.voltage_v:.2f}V"


SKY130_NOMINAL_LIBERTY_SUFFIX = "_tt_025C_1v80"
GF180MCU_3V3_NOMINAL_LIBERTY_SUFFIX = "_tt_025C_3v30"

# Verified Sky130 sky130_fd_sc_hd Liberty suffixes from
# efabless/skywater-pdk-libs-sky130_fd_sc_hd/timing @ 5a42fd01a0eeb2338340438443c7062e198d97a4
# collated into sky130A/libs.ref/sky130_fd_sc_hd/lib by open_pdks @ 0c1f82326278.
SKY130_LIBERTY_SUFFIXES: frozenset[str] = frozenset(
    (
        "_ff_100C_1v65",
        "_ff_100C_1v95",
        "_ff_n40C_1v56",
        "_ff_n40C_1v65",
        "_ff_n40C_1v76",
        "_ff_n40C_1v95",
        "_ff_n40C_1v95_ccsnoise",
        "_ss_100C_1v40",
        "_ss_100C_1v60",
        "_ss_n40C_1v28",
        "_ss_n40C_1v35",
        "_ss_n40C_1v40",
        "_ss_n40C_1v44",
        "_ss_n40C_1v60",
        "_ss_n40C_1v60_ccsnoise",
        "_ss_n40C_1v76",
        "_tt_025C_1v80",
        "_tt_100C_1v80",
    )
)

# Verified GF180MCU gf180mcu_fd_sc_mcu7t5v0 Liberty suffixes from
# google/globalfoundries-pdk-libs-gf180mcu_fd_sc_mcu7t5v0/liberty @ 43beb45e4d323a76
# and openroad/orfs@sha256:86dfae2d567b (flow/platforms/gf180/lib/).
GF180MCU_SUFFIXES_BY_NOMINAL: Dict[str, frozenset[str]] = {
    "_tt_025C_1v80": frozenset(
        (
            "_ff_125C_1v98",
            "_ff_n40C_1v98",
            "_ss_125C_1v62",
            "_ss_n40C_1v62",
            "_tt_025C_1v80",
        )
    ),
    "_tt_025C_3v30": frozenset(
        (
            "_ff_125C_3v60",
            "_ff_n40C_3v60",
            "_ss_125C_3v00",
            "_ss_n40C_3v00",
            "_tt_025C_3v30",
        )
    ),
    "_tt_025C_5v00": frozenset(
        (
            "_ff_125C_5v50",
            "_ff_n40C_5v50",
            "_ss_125C_4v50",
            "_ss_n40C_4v50",
            "_tt_025C_5v00",
        )
    ),
}

DEFAULT_CORNERS: List[PVTCorner] = [
    PVTCorner(CornerType.TT, 25.0, 1.80, "_tt_025C_1v80"),
    PVTCorner(CornerType.SS, 100.0, 1.60, "_ss_100C_1v60"),
    PVTCorner(CornerType.SS, -40.0, 1.60, "_ss_n40C_1v60"),
    PVTCorner(CornerType.FF, -40.0, 1.95, "_ff_n40C_1v95"),
    PVTCorner(CornerType.FF, 100.0, 1.95, "_ff_100C_1v95"),
]

_GF180MCU_3V3_CORNERS: List[PVTCorner] = [
    PVTCorner(CornerType.TT, 25.0, 3.30, "_tt_025C_3v30"),
    PVTCorner(CornerType.SS, 125.0, 3.00, "_ss_125C_3v00"),
    PVTCorner(CornerType.SS, -40.0, 3.00, "_ss_n40C_3v00"),
    PVTCorner(CornerType.FF, -40.0, 3.60, "_ff_n40C_3v60"),
    PVTCorner(CornerType.FF, 125.0, 3.60, "_ff_125C_3v60"),
]

_PDK_DEFAULT_CORNERS: Dict[str, List[PVTCorner]] = {
    "sky130": DEFAULT_CORNERS,
    "gf180mcu": _GF180MCU_3V3_CORNERS,
}
