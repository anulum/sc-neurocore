# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC signoff scripts, corners, derating, and summaries

"""Generate and evaluate ASIC timing, power, area, DRC, and LVS evidence."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sc_neurocore.asic_flow.design import DesignParams
from sc_neurocore.asic_flow.pdk import PDKConfig


@dataclass
class SignoffCheckResult:
    """Result of one signoff check."""

    check_name: str
    passed: bool
    details: str = ""
    metric: float = 0.0


class SignoffGenerator:
    """Generates signoff scripts and evaluates results."""

    @staticmethod
    def generate_sta_script(pdk: PDKConfig, design: DesignParams) -> str:
        """Generate OpenSTA timing analysis script."""
        return textwrap.dedent(f"""\
# SC-NeuroCore STA Signoff — OpenSTA
read_liberty {pdk.liberty_file}
read_verilog {design.top_module}_final.v
link_design {design.top_module}
read_sdc constraints_{design.top_module}.sdc

report_checks -path_delay min_max -format full_clock_expanded \\
    -fields {{slew cap input_pins nets}} \\
    -digits 4

report_tns
report_wns
report_power
""")

    @staticmethod
    def generate_drc_script(pdk: PDKConfig, design: DesignParams) -> str:
        """Generate DRC check script (KLayout-based for open PDKs)."""
        if pdk.is_open_source:
            return textwrap.dedent(f"""\
# SC-NeuroCore DRC — KLayout (open-source PDK)
import klayout.db as db
import klayout.rdb as rdb

layout = db.Layout()
layout.read("{design.top_module}.gds")
# Run DRC deck for {pdk.pdk_type.value}
# drc_deck = "$PDK_ROOT/{pdk.pdk_type.value}/libs.tech/klayout/drc/{pdk.pdk_type.value}.lydrc"
""")
        return f"# DRC for {pdk.pdk_type.value}: use vendor-specific tool\n"

    @staticmethod
    def generate_lvs_script(pdk: PDKConfig, design: DesignParams) -> str:
        """Generate LVS check script."""
        if pdk.is_open_source:
            return textwrap.dedent(f"""\
# SC-NeuroCore LVS — Netgen (open-source PDK)
netgen -batch lvs \\
    "{design.top_module}.spice {design.top_module}" \\
    "{design.top_module}_final.v {design.top_module}" \\
    $PDK_ROOT/{pdk.pdk_type.value}/libs.tech/netgen/{pdk.pdk_type.value}_setup.tcl \\
    lvs_{design.top_module}.log
""")
        return f"# LVS for {pdk.pdk_type.value}: use vendor-specific tool\n"

    @staticmethod
    def evaluate_timing(wns: float, tns: float, clock_period_ns: float) -> SignoffCheckResult:
        """Evaluate timing signoff from worst/total negative slack."""
        passed = wns >= 0.0
        details = f"WNS={wns:.3f}ns TNS={tns:.3f}ns period={clock_period_ns:.3f}ns"
        return SignoffCheckResult("STA", passed, details, wns)

    @staticmethod
    def evaluate_power(
        dynamic_mw: float, leakage_mw: float, budget_mw: float
    ) -> SignoffCheckResult:
        """Compare dynamic plus leakage power against a milliwatt budget."""
        total = dynamic_mw + leakage_mw
        passed = total <= budget_mw
        details = f"dynamic={dynamic_mw:.3f}mW leakage={leakage_mw:.3f}mW total={total:.3f}mW budget={budget_mw:.3f}mW"
        return SignoffCheckResult("Power", passed, details, total)

    @staticmethod
    def evaluate_area(
        cell_count: int, used_area_um2: float, die_area_um2: float
    ) -> SignoffCheckResult:
        """Compare placed-cell area with the 85 percent utilisation limit."""
        util = used_area_um2 / die_area_um2 if die_area_um2 > 0 else 0
        passed = util <= 0.85
        details = f"cells={cell_count} util={util:.1%} used={used_area_um2:.0f}µm² die={die_area_um2:.0f}µm²"
        return SignoffCheckResult("Area", passed, details, util)


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


DEFAULT_CORNERS = [
    PVTCorner(CornerType.TT, 25.0, 1.80, "_tt_025C_1v80"),
    PVTCorner(CornerType.SS, 125.0, 1.62, "_ss_125C_1v62"),
    PVTCorner(CornerType.FF, -40.0, 1.98, "_ff_n40C_1v98"),
    PVTCorner(CornerType.SF, 100.0, 1.62, "_sf_100C_1v62"),
    PVTCorner(CornerType.FS, -40.0, 1.98, "_fs_n40C_1v98"),
]


@dataclass
class MultiCornerAnalysis:
    """Generates multi-corner STA scripts for all PVT corners."""

    @staticmethod
    def generate(
        pdk: PDKConfig, design: DesignParams, corners: Optional[List[PVTCorner]] = None
    ) -> str:
        """Render one OpenSTA analysis section per selected PVT corner."""
        if corners is None:
            corners = DEFAULT_CORNERS
        lines = [f"# Multi-Corner STA for {design.top_module}"]
        for c in corners:
            lib = (
                pdk.liberty_file.replace("_tt_025C_1v80", c.liberty_suffix)
                if c.liberty_suffix
                else pdk.liberty_file
            )
            lines.append(f"\n# Corner: {c.label}")
            lines.append(f"read_liberty {lib}")
            lines.append(f"read_verilog {design.top_module}_final.v")
            lines.append(f"link_design {design.top_module}")
            lines.append(f"read_sdc constraints_{design.top_module}.sdc")
            lines.append("set_operating_conditions -analysis_type on_chip_variation")
            lines.append("report_checks -path_delay min_max -digits 4")
            lines.append("report_tns")
            lines.append("report_wns")
        return "\n".join(lines) + "\n"

    @staticmethod
    def worst_slack(per_corner_wns: Dict[str, float]) -> Tuple[str, float]:
        """Return the corner with the smallest worst negative slack."""
        if not per_corner_wns:
            return ("none", 0.0)
        worst = min(per_corner_wns.items(), key=lambda kv: kv[1])
        return worst


@dataclass
class OCVConfig:
    """On-Chip Variation derating factors."""

    data_cell_early: float = 0.95
    data_cell_late: float = 1.05
    data_net_early: float = 0.95
    data_net_late: float = 1.05
    clock_cell_early: float = 0.97
    clock_cell_late: float = 1.03

    def generate_sdc_fragment(self) -> str:
        """Render early and late cell/net derates as an SDC fragment."""
        return textwrap.dedent(f"""\
# OCV Derating
set_timing_derate -early {self.data_cell_early:.3f} -cell_delay [all_inputs]
set_timing_derate -late {self.data_cell_late:.3f} -cell_delay [all_inputs]
set_timing_derate -early {self.data_net_early:.3f} -net_delay [all_inputs]
set_timing_derate -late {self.data_net_late:.3f} -net_delay [all_inputs]
""")

    @classmethod
    def conservative(cls) -> OCVConfig:
        """Return wider early/late derates for screening runs."""
        return cls(
            data_cell_early=0.93,
            data_cell_late=1.07,
            data_net_early=0.93,
            data_net_late=1.07,
            clock_cell_early=0.95,
            clock_cell_late=1.05,
        )


@dataclass
class DRCViolation:
    """One DRC rule violation."""

    rule_name: str
    count: int
    severity: str = "error"


@dataclass
class SignoffSummary:
    """Structured signoff summary with pass/fail per check."""

    timing: SignoffCheckResult
    power: SignoffCheckResult
    area: SignoffCheckResult
    drc_violations: List[DRCViolation] = field(default_factory=list)
    lvs_match: bool = False

    @property
    def drc_clean(self) -> bool:
        """Return whether no counted error-severity DRC violation exists."""
        return not any(v.severity == "error" and v.count > 0 for v in self.drc_violations)

    @property
    def all_pass(self) -> bool:
        """Return whether timing, power, area, DRC, and LVS all pass."""
        return (
            self.timing.passed
            and self.power.passed
            and self.area.passed
            and self.drc_clean
            and self.lvs_match
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the signoff decision and counted DRC violations."""
        return {
            "timing": {"passed": self.timing.passed, "details": self.timing.details},
            "power": {"passed": self.power.passed, "details": self.power.details},
            "area": {"passed": self.area.passed, "details": self.area.details},
            "drc_clean": self.drc_clean,
            "drc_violations": [
                {"rule": v.rule_name, "count": v.count} for v in self.drc_violations
            ],
            "lvs_match": self.lvs_match,
            "all_pass": self.all_pass,
        }
