# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC signoff scripts, corners, derating, and summaries

"""Generate and evaluate ASIC timing, power, area, DRC, and LVS evidence."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sc_neurocore.asic_flow.design import DesignParams
from sc_neurocore.asic_flow.liberty_corners import (
    DEFAULT_CORNERS as DEFAULT_CORNERS,
    GF180MCU_SUFFIXES_BY_NOMINAL,
    SKY130_LIBERTY_SUFFIXES,
    _PDK_DEFAULT_CORNERS,
    CornerType as CornerType,
    PVTCorner as PVTCorner,
)
from sc_neurocore.asic_flow.pdk import PDKConfig, PDKType

_INFERRED_NOMINAL_TOKEN_RE = re.compile(r"(_(?:tt|typ)_[A-Za-z0-9_]+)(?=\.lib$)")


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


@dataclass
class MultiCornerAnalysis:
    """Generates multi-corner STA scripts for verified PVT corners."""

    @staticmethod
    def _corner_liberty_path(pdk: PDKConfig, liberty_suffix: str) -> str:
        """Resolve one corner's Liberty path from ``pdk`` and ``liberty_suffix``."""
        if not liberty_suffix:
            return pdk.liberty_file
        allowed: Optional[frozenset[str]] = None
        if pdk.pdk_type == PDKType.SKY130:
            allowed = SKY130_LIBERTY_SUFFIXES
        elif pdk.pdk_type == PDKType.GF180MCU:
            allowed = GF180MCU_SUFFIXES_BY_NOMINAL.get(pdk.nominal_liberty_suffix)
            if allowed is None:
                raise ValueError(
                    f"unknown gf180mcu nominal_liberty_suffix "
                    f"{pdk.nominal_liberty_suffix!r}; expected one of "
                    f"{sorted(GF180MCU_SUFFIXES_BY_NOMINAL)}"
                )
        if allowed is not None and liberty_suffix not in allowed:
            raise ValueError(
                f"corner suffix {liberty_suffix!r} is not in the verified "
                f"{pdk.pdk_type.value} Liberty table"
            )
        if not pdk.liberty_file:
            raise ValueError("cannot substitute corner suffix on an empty liberty_file")
        parent, sep, filename = pdk.liberty_file.rpartition("/")
        token = pdk.nominal_liberty_suffix
        if not token:
            match = _INFERRED_NOMINAL_TOKEN_RE.search(filename)
            if match is not None:
                token = match.group(1)
        token_pos = filename.rfind(token) if token else -1
        if token_pos < 0:
            raise ValueError(
                f"nominal_liberty_suffix {token!r} not found in liberty_file {filename!r}"
            )
        replaced = filename[:token_pos] + liberty_suffix + filename[token_pos + len(token) :]
        return f"{parent}{sep}{replaced}"

    @staticmethod
    def generate(
        pdk: PDKConfig, design: DesignParams, corners: Optional[List[PVTCorner]] = None
    ) -> str:
        """Render one OpenSTA analysis section per selected PVT corner."""
        if corners is None:
            table = _PDK_DEFAULT_CORNERS.get(pdk.pdk_type.value)
            if table is None:
                raise ValueError(
                    f"no verified PVT corner table for PDK {pdk.pdk_type.value!r}; "
                    "pass explicit corners"
                )
            corners = table
        lines = [f"# Multi-Corner STA for {design.top_module}"]
        for c in corners:
            lib = MultiCornerAnalysis._corner_liberty_path(pdk, c.liberty_suffix)
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
        return min(per_corner_wns.items(), key=lambda kv: kv[1])


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
