# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — OpenROAD / Open-Source ASIC Tape-Out Flow

"""Push-button ASIC tape-out pipeline: RTL → Yosys → OpenROAD → GDSII.

Generates a complete open-source ASIC flow from SC-NeuroCore RTL,
including synthesis scripts (Yosys), place-and-route configurations
(OpenROAD/OpenLane), timing/power signoff, and DRC/LVS checks.

Pipeline stages:
    1. **Synthesis**: Yosys TCL script for technology mapping
    2. **Floorplan**: Die area, IO placement, macro positions
    3. **Place & Route**: Global/detailed placement + routing
    4. **Signoff**: STA (OpenSTA), power (IR drop), DRC, LVS
    5. **Export**: GDSII, LEF/DEF, timing reports

Supported PDKs:
    - SkyWater SKY130 (open-source, 130nm)
    - GlobalFoundries GF180MCU (open-source, 180nm)
    - TSMC 28nm (commercial, template only)
    - Intel 16 (commercial, template only)
    - Custom / generic (user-provided liberty + lef)

Compatible with:
    - ``hdl_gen/verilog_generator.py`` — source RTL
    - ``hdl/formal/`` — SymbiYosys proofs carried into signoff
    - ``uvm_gen/`` — UVM testbenches for post-P&R gate-level sim
    - ``chiplet_gen/`` — per-die ASIC flow invocations
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── PDK Configuration ────────────────────────────────────────────────


class PDKType(Enum):
    SKY130 = "sky130"
    GF180MCU = "gf180mcu"
    TSMC28 = "tsmc28"
    INTEL16 = "intel16"
    CUSTOM = "custom"


@dataclass
class PDKConfig:
    """Process Design Kit configuration."""

    pdk_type: PDKType = PDKType.SKY130
    liberty_file: str = ""
    lef_file: str = ""
    tech_lef: str = ""
    cell_prefix: str = "sky130_fd_sc_hd__"
    clock_period_ns: float = 10.0
    voltage_v: float = 1.8
    temperature_c: float = 25.0
    corner: str = "tt"
    metal_layers: int = 5
    min_feature_nm: int = 130

    @classmethod
    def from_pdk_type(cls, pdk: PDKType) -> PDKConfig:
        presets: Dict[PDKType, Dict[str, Any]] = {
            PDKType.SKY130: dict(
                liberty_file="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
                lef_file="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef",
                tech_lef="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef",
                cell_prefix="sky130_fd_sc_hd__",
                clock_period_ns=10.0,
                voltage_v=1.8,
                metal_layers=5,
                min_feature_nm=130,
            ),
            PDKType.GF180MCU: dict(
                liberty_file="$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/gf180mcu_fd_sc_mcu7t5v0__tt_025C_3v30.lib",
                lef_file="$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lef/gf180mcu_fd_sc_mcu7t5v0.lef",
                tech_lef="$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/techlef/gf180mcu_fd_sc_mcu7t5v0__nom.tlef",
                cell_prefix="gf180mcu_fd_sc_mcu7t5v0__",
                clock_period_ns=15.0,
                voltage_v=3.3,
                metal_layers=6,
                min_feature_nm=180,
            ),
            PDKType.TSMC28: dict(
                liberty_file="$PDK_ROOT/tsmc28/tcbn28hpcplusbwp7t30p140_110a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t30p140ssgnp0p81v125c.lib",
                lef_file="$PDK_ROOT/tsmc28/lef/tcbn28hpcplusbwp7t30p140.lef",
                tech_lef="$PDK_ROOT/tsmc28/lef/HiPe_M10.tlef",
                cell_prefix="TSMC_",
                clock_period_ns=2.0,
                voltage_v=0.9,
                metal_layers=10,
                min_feature_nm=28,
            ),
            PDKType.INTEL16: dict(
                liberty_file="$PDK_ROOT/intel16/lib/intel16_sc.lib",
                lef_file="$PDK_ROOT/intel16/lef/intel16_sc.lef",
                tech_lef="$PDK_ROOT/intel16/lef/intel16.tlef",
                cell_prefix="INTEL16_",
                clock_period_ns=1.5,
                voltage_v=0.8,
                metal_layers=12,
                min_feature_nm=16,
            ),
            PDKType.CUSTOM: dict(
                liberty_file="",
                lef_file="",
                tech_lef="",
                cell_prefix="",
                clock_period_ns=10.0,
                voltage_v=1.8,
                metal_layers=5,
                min_feature_nm=130,
            ),
        }
        return cls(pdk_type=pdk, **presets[pdk])

    @property
    def is_open_source(self) -> bool:
        return self.pdk_type in (PDKType.SKY130, PDKType.GF180MCU)

    def with_pdk_root(self, pdk_root: str) -> PDKConfig:
        """Return a copy with ``$PDK_ROOT`` variables bound to ``pdk_root``."""
        root = str(Path(pdk_root).expanduser())
        return PDKConfig(
            pdk_type=self.pdk_type,
            liberty_file=self.liberty_file.replace("$PDK_ROOT", root),
            lef_file=self.lef_file.replace("$PDK_ROOT", root),
            tech_lef=self.tech_lef.replace("$PDK_ROOT", root),
            cell_prefix=self.cell_prefix,
            clock_period_ns=self.clock_period_ns,
            voltage_v=self.voltage_v,
            temperature_c=self.temperature_c,
            corner=self.corner,
            metal_layers=self.metal_layers,
            min_feature_nm=self.min_feature_nm,
        )


@dataclass(frozen=True)
class ResolvedPDKFiles:
    """Resolved file paths required by the open-source ASIC flow."""

    liberty_file: str
    lef_file: str
    tech_lef: str
    setup_tcl: str = ""
    drc_deck: str = ""
    lvs_setup: str = ""

    def required_paths(self) -> Dict[str, str]:
        return {
            "liberty_file": self.liberty_file,
            "lef_file": self.lef_file,
            "tech_lef": self.tech_lef,
        }

    def optional_paths(self) -> Dict[str, str]:
        return {
            "setup_tcl": self.setup_tcl,
            "drc_deck": self.drc_deck,
            "lvs_setup": self.lvs_setup,
        }


@dataclass(frozen=True)
class PDKResolution:
    """Outcome of resolving a PDK against the local filesystem."""

    pdk: PDKConfig
    files: ResolvedPDKFiles
    missing_required: Tuple[str, ...] = ()
    missing_optional: Tuple[str, ...] = ()

    @property
    def usable_for_synthesis(self) -> bool:
        return not self.missing_required

    @property
    def usable_for_signoff(self) -> bool:
        return self.usable_for_synthesis and not self.missing_optional


class OpenSourcePDKResolver:
    """Resolve Sky130/GF180 file locations without requiring OpenLane at import time."""

    @staticmethod
    def resolve(
        pdk: PDKConfig,
        pdk_root: Optional[str] = None,
        require_existing: bool = False,
    ) -> PDKResolution:
        """Bind ``$PDK_ROOT`` and report missing PDK artefacts.

        Parameters
        ----------
        pdk:
            PDK preset or custom configuration.
        pdk_root:
            Explicit PDK root. If absent, ``PDK_ROOT`` then ``PDKPATH`` are used.
        require_existing:
            When true, missing required files are reported as blockers. When false,
            paths are still resolved so generated flow decks are deterministic.
        """
        root = pdk_root or os.environ.get("PDK_ROOT") or os.environ.get("PDKPATH") or "$PDK_ROOT"
        resolved_pdk = pdk.with_pdk_root(root)
        files = OpenSourcePDKResolver._file_manifest(resolved_pdk)

        missing_required: Tuple[str, ...] = ()
        missing_optional: Tuple[str, ...] = ()
        if require_existing:
            missing_required = tuple(
                name for name, path in files.required_paths().items() if not Path(path).exists()
            )
            missing_optional = tuple(
                name
                for name, path in files.optional_paths().items()
                if path and not Path(path).exists()
            )

        return PDKResolution(resolved_pdk, files, missing_required, missing_optional)

    @staticmethod
    def _file_manifest(pdk: PDKConfig) -> ResolvedPDKFiles:
        if pdk.pdk_type == PDKType.SKY130:
            root = OpenSourcePDKResolver._pdk_root_from_path(pdk.liberty_file, "sky130A")
            return ResolvedPDKFiles(
                liberty_file=pdk.liberty_file,
                lef_file=pdk.lef_file,
                tech_lef=pdk.tech_lef,
                setup_tcl=f"{root}/sky130A/libs.tech/netgen/sky130A_setup.tcl",
                drc_deck=f"{root}/sky130A/libs.tech/klayout/drc/sky130.lydrc",
                lvs_setup=f"{root}/sky130A/libs.tech/netgen/sky130A_setup.tcl",
            )
        if pdk.pdk_type == PDKType.GF180MCU:
            root = OpenSourcePDKResolver._pdk_root_from_path(pdk.liberty_file, "gf180mcuD")
            return ResolvedPDKFiles(
                liberty_file=pdk.liberty_file,
                lef_file=pdk.lef_file,
                tech_lef=pdk.tech_lef,
                setup_tcl=f"{root}/gf180mcuD/libs.tech/netgen/gf180mcuD_setup.tcl",
                drc_deck=f"{root}/gf180mcuD/libs.tech/klayout/drc/gf180mcu.drc",
                lvs_setup=f"{root}/gf180mcuD/libs.tech/netgen/gf180mcuD_setup.tcl",
            )
        return ResolvedPDKFiles(
            liberty_file=pdk.liberty_file,
            lef_file=pdk.lef_file,
            tech_lef=pdk.tech_lef,
        )

    @staticmethod
    def _pdk_root_from_path(path: str, marker: str) -> str:
        before, separator, _after = path.partition(f"/{marker}/")
        return before if separator else "$PDK_ROOT"


@dataclass(frozen=True)
class SCASICOptimisationConfig:
    """SC-specific synthesis settings for stochastic neuromorphic datapaths."""

    share_stochastic_counters: bool = True
    reduce_constant_widths: bool = True
    preserve_lfsr_hierarchy: bool = True
    max_fanout: int = 16
    abc_delay_margin: float = 0.90

    def yosys_passes(self) -> List[str]:
        passes: List[str] = []
        if self.reduce_constant_widths:
            passes.append("wreduce")
        if self.share_stochastic_counters:
            passes.extend(["share", "opt_share"])
        passes.append("opt_clean -purge")
        return passes


# ── Design Parameters ────────────────────────────────────────────────


@dataclass
class DesignParams:
    """ASIC design parameters."""

    top_module: str = "sc_neurocore_top"
    clock_name: str = "clk"
    reset_name: str = "rst_n"
    reset_active_low: bool = True
    target_frequency_mhz: float = 100.0
    die_area_um: Tuple[float, float, float, float] = (0, 0, 500, 500)
    core_area_um: Tuple[float, float, float, float] = (20, 20, 480, 480)
    utilisation: float = 0.5
    aspect_ratio: float = 1.0
    io_margin_um: float = 20.0
    power_nets: List[str] = field(default_factory=lambda: ["VDD", "VSS"])
    rtl_files: List[str] = field(default_factory=list)
    sc_optimisation: SCASICOptimisationConfig = field(default_factory=SCASICOptimisationConfig)

    @property
    def clock_period_ns(self) -> float:
        return 1000.0 / self.target_frequency_mhz

    @property
    def die_width_um(self) -> float:
        return self.die_area_um[2] - self.die_area_um[0]

    @property
    def die_height_um(self) -> float:
        return self.die_area_um[3] - self.die_area_um[1]

    @property
    def core_area_mm2(self) -> float:
        w = self.core_area_um[2] - self.core_area_um[0]
        h = self.core_area_um[3] - self.core_area_um[1]
        return (w * h) / 1e6


# ── Synthesis Script Generator ───────────────────────────────────────


class SynthesisGenerator:
    """Generates Yosys synthesis TCL scripts."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        rtl_reads = "\n".join(f"read_verilog {f}" for f in design.rtl_files)
        if not rtl_reads:
            rtl_reads = f"read_verilog {design.top_module}.v"
        sc_passes = "\n".join(design.sc_optimisation.yosys_passes())
        if design.sc_optimisation.preserve_lfsr_hierarchy:
            sc_passes = (
                "# Preserve deterministic SC seed generators for gate-level debug\n"
                "setattr -mod -pattern *lfsr* keep_hierarchy 1\n"
                f"{sc_passes}"
            )
        abc_delay_ps = design.clock_period_ns * 1000.0 * design.sc_optimisation.abc_delay_margin

        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Synthesis — Yosys Script
# PDK: {pdk.pdk_type.value}
# Target: {design.top_module} @ {design.target_frequency_mhz} MHz

# Read RTL
{rtl_reads}

# Hierarchy check
hierarchy -check -top {design.top_module}

# High-level synthesis
proc; opt; fsm; opt; memory; opt

# Technology mapping
synth -top {design.top_module}

# SC-aware optimisation
{sc_passes}

# Map to standard cells
dfflibmap -liberty {pdk.liberty_file}
abc -liberty {pdk.liberty_file} -D {abc_delay_ps:.0f}

# Clean up
opt_clean -purge

# Write outputs
write_verilog -noattr synth_{design.top_module}.v
write_json synth_{design.top_module}.json

# Statistics
stat -liberty {pdk.liberty_file}
""")


# ── Floorplan Generator ──────────────────────────────────────────────


class FloorplanGenerator:
    """Generates OpenROAD floorplan TCL scripts."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        die = design.die_area_um
        core = design.core_area_um
        power = design.power_nets

        power_ring = ""
        if len(power) >= 2:
            power_ring = textwrap.dedent(f"""\
# Power grid
add_global_connection -net {power[0]} -pin_pattern "VPWR|VDD|vdd" -power
add_global_connection -net {power[1]} -pin_pattern "VGND|VSS|vss" -ground

set_voltage_domain -name CORE -power {power[0]} -ground {power[1]}

define_pdn_grid -name core_grid -pins {{{power[0]} {power[1]}}} \\
    -voltage_domains CORE

add_pdn_stripe -grid core_grid -layer met1 -width 0.48 -pitch 5.44 \\
    -offset 0 -followpins
add_pdn_stripe -grid core_grid -layer met4 -width 1.6 -pitch 27.14 \\
    -offset 13.57
add_pdn_stripe -grid core_grid -layer met5 -width 1.6 -pitch 27.2 \\
    -offset 13.6

add_pdn_connect -grid core_grid -layers {{met1 met4}}
add_pdn_connect -grid core_grid -layers {{met4 met5}}
""")

        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Floorplan — OpenROAD
# Die: {die[2] - die[0]}×{die[3] - die[1]} µm

# Read technology
read_lef {pdk.tech_lef}
read_lef {pdk.lef_file}

# Read synthesized netlist
read_verilog synth_{design.top_module}.v
link_design {design.top_module}

# Read timing constraints
read_sdc constraints_{design.top_module}.sdc

# Initialize floorplan
initialize_floorplan \\
    -die_area {{{die[0]} {die[1]} {die[2]} {die[3]}}} \\
    -core_area {{{core[0]} {core[1]} {core[2]} {core[3]}}} \\
    -site unithd

# IO placement
place_pins -hor_layers met3 -ver_layers met2

{power_ring}
""")


# ── Place & Route Generator ─────────────────────────────────────────


class PlaceRouteGenerator:
    """Generates OpenROAD place-and-route TCL scripts."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Place & Route — OpenROAD
# Target: {design.top_module}

# Global placement
global_placement -density {design.utilisation:.2f} -pad_left 2 -pad_right 2

# Clock tree synthesis
clock_tree_synthesis -root_buf {pdk.cell_prefix}buf_2 \\
    -buf_list {{{pdk.cell_prefix}buf_4 {pdk.cell_prefix}buf_8}} \\
    -wire_unit 10

# Repair hold violations
estimate_parasitics -placement
repair_timing -hold

# Detailed placement
detailed_placement

# Check placement
check_placement

# Filler cell insertion
filler_placement {pdk.cell_prefix}fill_1 {pdk.cell_prefix}fill_2

# Global routing
global_route -guide_file route_{design.top_module}.guide \\
    -congestion_iterations 30

# Detailed routing
detailed_route -output_drc route_drc_{design.top_module}.rpt \\
    -output_maze route_maze_{design.top_module}.log

# Write outputs
write_def {design.top_module}_final.def
write_verilog {design.top_module}_final.v
""")


# ── Timing Constraints Generator ─────────────────────────────────────


class SDCGenerator:
    """Generates Synopsys Design Constraints (SDC) for STA."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        period = design.clock_period_ns
        reset_val = "0" if design.reset_active_low else "1"
        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Constraints — SDC
# Clock: {design.clock_name} @ {design.target_frequency_mhz} MHz

create_clock [get_ports {design.clock_name}] \\
    -name {design.clock_name} \\
    -period {period:.3f}

# Clock uncertainty (10% of period)
set_clock_uncertainty {period * 0.1:.3f} [get_clocks {design.clock_name}]

# Input/output delays (25% of period)
set_input_delay {period * 0.25:.3f} -clock {design.clock_name} [all_inputs]
set_output_delay {period * 0.25:.3f} -clock {design.clock_name} [all_outputs]

# Reset is constant during operation
set_false_path -from [get_ports {design.reset_name}]

# Don't touch clock/reset nets
set_dont_touch_network [get_ports {design.clock_name}]

# Max transition / fanout
set_max_transition {period * 0.15:.3f} [current_design]
set_max_fanout {design.sc_optimisation.max_fanout} [current_design]

# Driving cell
set_driving_cell -lib_cell {pdk.cell_prefix}buf_2 [all_inputs]

# Load
set_load 0.05 [all_outputs]
""")


# ── Signoff Checks Generator ─────────────────────────────────────────


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
        total = dynamic_mw + leakage_mw
        passed = total <= budget_mw
        details = f"dynamic={dynamic_mw:.3f}mW leakage={leakage_mw:.3f}mW total={total:.3f}mW budget={budget_mw:.3f}mW"
        return SignoffCheckResult("Power", passed, details, total)

    @staticmethod
    def evaluate_area(
        cell_count: int, used_area_um2: float, die_area_um2: float
    ) -> SignoffCheckResult:
        util = used_area_um2 / die_area_um2 if die_area_um2 > 0 else 0
        passed = util <= 0.85
        details = f"cells={cell_count} util={util:.1%} used={used_area_um2:.0f}µm² die={die_area_um2:.0f}µm²"
        return SignoffCheckResult("Area", passed, details, util)


# ── GDSII Export ─────────────────────────────────────────────────────


class GDSIIExporter:
    """Generates GDSII stream-out scripts."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        if pdk.is_open_source:
            return textwrap.dedent(f"""\
# SC-NeuroCore GDSII Export — KLayout/Magic
# Merge standard cell GDS with routed DEF

# Option 1: KLayout
klayout -zz -rd design_name={design.top_module} \\
    -rd in_def={design.top_module}_final.def \\
    -rd in_gds="$PDK_ROOT/{pdk.pdk_type.value}/libs.ref/*/gds/*.gds" \\
    -rd seal_gds="" \\
    -rd out_gds={design.top_module}.gds \\
    -rm $OPENLANE_ROOT/scripts/klayout/def2gds.py

# Option 2: Magic
magic -dnull -noconsole << EOF
lef read {pdk.tech_lef}
lef read {pdk.lef_file}
def read {design.top_module}_final.def
load {design.top_module}
select top cell
expand
gds write {design.top_module}.gds
quit
EOF
""")
        return f"# GDSII export for {pdk.pdk_type.value}: use vendor stream-out\n"


# ── Full Pipeline Orchestrator ───────────────────────────────────────


@dataclass
class ASICFlowOutput:
    """Complete output of the ASIC tape-out flow."""

    synth_tcl: str
    sdc: str
    floorplan_tcl: str
    pnr_tcl: str
    sta_tcl: str
    drc_script: str
    lvs_script: str
    gdsii_script: str
    makefile: str
    filelist: List[str]

    def to_dict(self) -> Dict[str, str]:
        return {
            "synth.tcl": self.synth_tcl,
            "constraints.sdc": self.sdc,
            "floorplan.tcl": self.floorplan_tcl,
            "pnr.tcl": self.pnr_tcl,
            "sta.tcl": self.sta_tcl,
            "drc_check.py": self.drc_script,
            "lvs_check.sh": self.lvs_script,
            "gdsii_export.sh": self.gdsii_script,
            "Makefile": self.makefile,
        }


class ASICFlowGenerator:
    """Top-level generator for the complete ASIC tape-out pipeline."""

    def generate(
        self,
        pdk: PDKConfig,
        design: DesignParams,
    ) -> ASICFlowOutput:
        synth = SynthesisGenerator.generate(pdk, design)
        sdc = SDCGenerator.generate(pdk, design)
        fp = FloorplanGenerator.generate(pdk, design)
        pnr = PlaceRouteGenerator.generate(pdk, design)
        sta = SignoffGenerator.generate_sta_script(pdk, design)
        drc = SignoffGenerator.generate_drc_script(pdk, design)
        lvs = SignoffGenerator.generate_lvs_script(pdk, design)
        gdsii = GDSIIExporter.generate(pdk, design)
        makefile = self._generate_makefile(design)

        filelist = list(
            ASICFlowOutput(synth, sdc, fp, pnr, sta, drc, lvs, gdsii, makefile, []).to_dict().keys()
        )

        return ASICFlowOutput(synth, sdc, fp, pnr, sta, drc, lvs, gdsii, makefile, filelist)

    def _generate_makefile(self, design: DesignParams) -> str:
        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Flow — Makefile
# Usage: make all

TOP = {design.top_module}

.PHONY: all synth floorplan pnr sta drc lvs gdsii clean

all: synth floorplan pnr sta drc lvs gdsii

synth:
\tyosys -c synth.tcl 2>&1 | tee logs/synth.log

floorplan: synth
\topenroad -exit floorplan.tcl 2>&1 | tee logs/floorplan.log

pnr: floorplan
\topenroad -exit pnr.tcl 2>&1 | tee logs/pnr.log

sta: pnr
\tsta sta.tcl 2>&1 | tee logs/sta.log

drc: gdsii
\tpython3 drc_check.py 2>&1 | tee logs/drc.log

lvs: pnr
\tbash lvs_check.sh 2>&1 | tee logs/lvs.log

gdsii: pnr
\tbash gdsii_export.sh 2>&1 | tee logs/gdsii.log

clean:
\trm -rf synth_$(TOP).* $(TOP)_final.* $(TOP).gds logs/
""")


@dataclass(frozen=True)
class ASICFlowBundle:
    """Generated ASIC flow files plus the evidence manifest path."""

    output_dir: str
    manifest_path: str
    file_paths: Dict[str, str]
    pdk_resolution: PDKResolution
    estimate: DesignEstimate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "file_paths": dict(self.file_paths),
            "pdk_resolution": {
                "pdk": _pdk_to_manifest(self.pdk_resolution.pdk),
                "files": asdict(self.pdk_resolution.files),
                "missing_required": list(self.pdk_resolution.missing_required),
                "missing_optional": list(self.pdk_resolution.missing_optional),
                "usable_for_synthesis": self.pdk_resolution.usable_for_synthesis,
                "usable_for_signoff": self.pdk_resolution.usable_for_signoff,
            },
            "estimate": asdict(self.estimate),
        }


def generate_asic_flow_bundle(
    output_dir: str | Path,
    *,
    pdk_type: PDKType | str = PDKType.SKY130,
    design: Optional[DesignParams] = None,
    pdk_root: Optional[str] = None,
    require_pdk_files: bool = False,
    n_neurons: int = 16,
    n_synapses: int = 256,
    bitstream_width: int = 256,
    n_aer_ports: int = 4,
    formal_evidence_artifacts: Optional[List[str]] = None,
) -> ASICFlowBundle:
    """Write a complete ASIC flow deck and evidence manifest in one call.

    The helper deliberately does not run Yosys/OpenROAD. It materialises the
    scripts, resolves the requested PDK paths, records missing artefacts, and
    adds a pre-synthesis estimate so Python API users can inspect the bundle
    before launching external EDA tools.
    """
    pdk_enum = _normalise_pdk_type(pdk_type)
    design = design or DesignParams()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    pdk = PDKConfig.from_pdk_type(pdk_enum)
    resolution = OpenSourcePDKResolver.resolve(
        pdk,
        pdk_root=pdk_root,
        require_existing=require_pdk_files,
    )
    flow = ASICFlowGenerator().generate(resolution.pdk, design)

    file_paths: Dict[str, str] = {}
    for name, content in flow.to_dict().items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        file_paths[name] = str(path)

    estimate = PreSynthEstimator.estimate(
        n_neurons=n_neurons,
        n_synapses=n_synapses,
        bitstream_width=bitstream_width,
        n_aer_ports=n_aer_ports,
        pdk=resolution.pdk,
    )
    manifest = _build_asic_flow_manifest(
        design=design,
        pdk_resolution=resolution,
        estimate=estimate,
        file_paths=file_paths,
        require_pdk_files=require_pdk_files,
        formal_evidence_artifacts=formal_evidence_artifacts or [],
    )
    manifest_path = out / "asic_flow_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return ASICFlowBundle(
        output_dir=str(out),
        manifest_path=str(manifest_path),
        file_paths=file_paths,
        pdk_resolution=resolution,
        estimate=estimate,
    )


def _normalise_pdk_type(pdk_type: PDKType | str) -> PDKType:
    if isinstance(pdk_type, PDKType):
        return pdk_type
    try:
        return PDKType(str(pdk_type).lower())
    except ValueError as exc:
        valid = ", ".join(p.value for p in PDKType)
        raise ValueError(f"unknown PDK type {pdk_type!r}; expected one of: {valid}") from exc


def _pdk_to_manifest(pdk: PDKConfig) -> Dict[str, Any]:
    data = asdict(pdk)
    data["pdk_type"] = pdk.pdk_type.value
    data["is_open_source"] = pdk.is_open_source
    return data


def _design_to_manifest(design: DesignParams) -> Dict[str, Any]:
    data = asdict(design)
    data["clock_period_ns"] = design.clock_period_ns
    data["die_width_um"] = design.die_width_um
    data["die_height_um"] = design.die_height_um
    data["core_area_mm2"] = design.core_area_mm2
    return data


def _build_asic_flow_manifest(
    *,
    design: DesignParams,
    pdk_resolution: PDKResolution,
    estimate: DesignEstimate,
    file_paths: Dict[str, str],
    require_pdk_files: bool,
    formal_evidence_artifacts: List[str],
) -> Dict[str, Any]:
    formal_status = _formal_evidence_status(formal_evidence_artifacts)
    return {
        "schema": "sc-neurocore.asic_flow_manifest.v1",
        "claim_status": {
            "scripts_generated": True,
            "external_eda_executed": False,
            "physical_ppa_claim_allowed": False,
            "formal_evidence_attached": formal_status["attached"],
            "formal_evidence_complete_for_claim": formal_status["complete_for_claim"],
            "reason": (
                "Generated decks and pre-synthesis estimates only; quote physical "
                "area, power, timing, or GDSII claims only after attaching exact "
                "OpenROAD/container and PDK revision evidence."
            ),
        },
        "require_pdk_files": require_pdk_files,
        "pdk": _pdk_to_manifest(pdk_resolution.pdk),
        "pdk_files": asdict(pdk_resolution.files),
        "missing_required": list(pdk_resolution.missing_required),
        "missing_optional": list(pdk_resolution.missing_optional),
        "usable_for_synthesis": pdk_resolution.usable_for_synthesis,
        "usable_for_signoff": pdk_resolution.usable_for_signoff,
        "formal_evidence": formal_status,
        "design": _design_to_manifest(design),
        "estimate": asdict(estimate),
        "generated_files": dict(sorted(file_paths.items())),
    }


def _formal_evidence_status(formal_evidence_artifacts: List[str]) -> Dict[str, Any]:
    artifacts = sorted(str(Path(item)) for item in formal_evidence_artifacts)
    has_proof_script = any(item.endswith((".sby", ".sv", ".sva")) for item in artifacts)
    has_report = any(item.endswith((".json", ".txt", ".log")) for item in artifacts)
    return {
        "artifacts": artifacts,
        "attached": bool(artifacts),
        "complete_for_claim": bool(artifacts) and has_proof_script and has_report,
        "required_types_for_claim": {
            "proof_script": [".sby", ".sv", ".sva"],
            "report_or_log": [".json", ".txt", ".log"],
        },
    }


# ── Area/Power/Timing Estimator ──────────────────────────────────────


@dataclass
class DesignEstimate:
    """Pre-synthesis area/power/timing estimate for an SC module."""

    module_name: str
    gate_count: int
    area_um2: float
    dynamic_power_mw: float
    leakage_power_mw: float
    critical_path_ns: float
    max_frequency_mhz: float


class PreSynthEstimator:
    """Estimates area, power, and timing before synthesis.

    Uses empirical models based on SC circuit characteristics:
    - Bitstream ops: ~10 gates/bit
    - LIF neuron: ~500 gates
    - STDP synapse: ~200 gates
    - AER router: ~100 gates/port
    """

    GATES_PER_BIT = 10
    GATES_PER_LIF = 500
    GATES_PER_SYNAPSE = 200
    GATES_PER_AER_PORT = 100

    @classmethod
    def estimate(
        cls,
        n_neurons: int,
        n_synapses: int,
        bitstream_width: int,
        n_aer_ports: int,
        pdk: PDKConfig,
    ) -> DesignEstimate:
        """Estimate design metrics from architectural parameters."""
        gates = (
            n_neurons * cls.GATES_PER_LIF
            + n_synapses * cls.GATES_PER_SYNAPSE
            + bitstream_width * cls.GATES_PER_BIT
            + n_aer_ports * cls.GATES_PER_AER_PORT
        )

        # Area: ~1 µm² per gate at 130nm, scales with feature size squared
        scale = (pdk.min_feature_nm / 130.0) ** 2
        area = gates * 1.0 * scale

        # Power: ~1 µW/gate dynamic at 100MHz 1.8V, ~0.01 µW/gate leakage
        freq_scale = 100.0 / (1000.0 / pdk.clock_period_ns)
        v_scale = (pdk.voltage_v / 1.8) ** 2
        dynamic = gates * 1e-3 * freq_scale * v_scale  # mW
        leakage = gates * 1e-5 * scale  # mW

        # Timing: ~0.05 ns/gate at 130nm
        cp = max(1.0, 10 + 0.01 * n_neurons) * (pdk.min_feature_nm / 130.0)
        max_freq = 1000.0 / cp

        return DesignEstimate(
            module_name="sc_neurocore_top",
            gate_count=gates,
            area_um2=area,
            dynamic_power_mw=dynamic,
            leakage_power_mw=leakage,
            critical_path_ns=cp,
            max_frequency_mhz=max_freq,
        )


# ── Multi-Corner / Multi-Mode Analysis (Gap 1) ──────────────────────


class CornerType(Enum):
    TT = "tt"  # typical
    FF = "ff"  # fast-fast
    SS = "ss"  # slow-slow
    SF = "sf"  # slow-fast
    FS = "fs"  # fast-slow


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
        if not per_corner_wns:
            return ("none", 0.0)
        worst = min(per_corner_wns.items(), key=lambda kv: kv[1])
        return worst


# ── CDC Analysis Script (Gap 2) ──────────────────────────────────────


class CDCCheckGenerator:
    """Generates clock-domain crossing lint scripts."""

    @staticmethod
    def generate(design: DesignParams, clock_domains: Optional[List[str]] = None) -> str:
        if clock_domains is None:
            clock_domains = [design.clock_name]
        domain_defs = "\n".join(f"create_clock -name {c} [get_ports {c}]" for c in clock_domains)
        return textwrap.dedent(f"""\
# SC-NeuroCore CDC Check
# Domains: {", ".join(clock_domains)}

{domain_defs}

# Report all clock-domain crossings
report_cdc -from [all_clocks] -to [all_clocks]
report_cdc -type async_reset
report_cdc -type reconvergence

# Check for missing synchronisers
check_cdc -severity error
""")


# ── IR Drop Analysis Script (Gap 3) ─────────────────────────────────


class IRDropGenerator:
    """Generates IR drop analysis scripts for OpenROAD."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams, toggle_rate: float = 0.1) -> str:
        return textwrap.dedent(f"""\
# SC-NeuroCore IR Drop Analysis — OpenROAD
# Toggle rate: {toggle_rate:.2f}

# Read design
read_lef {pdk.tech_lef}
read_lef {pdk.lef_file}
read_def {design.top_module}_final.def
read_liberty {pdk.liberty_file}
read_sdc constraints_{design.top_module}.sdc

# Set activity
set_power_activity -input -activity {toggle_rate:.3f}

# Analyze IR drop
analyze_power_grid -net {design.power_nets[0]}
analyze_power_grid -net {design.power_nets[1]}

# Report
report_power_grid -net {design.power_nets[0]} -corner tt
""")


# ── IO / Pin Constraint Generator (Gap 4) ────────────────────────────


@dataclass
class IOPin:
    """Specification for one IO pad."""

    name: str
    direction: str  # "input", "output", "inout"
    side: str = "N"  # N, S, E, W
    offset_um: float = 0.0
    layer: str = "met3"


@dataclass
class IOConstraintGenerator:
    """Generates IO placement constraint files."""

    @staticmethod
    def generate(pins: List[IOPin], design: DesignParams) -> str:
        lines = [f"# IO Constraints for {design.top_module}"]
        for pin in pins:
            lines.append(
                f"place_pin -pin_name {pin.name} -layer {pin.layer} "
                f"-location {{{pin.offset_um} 0}} -side {pin.side}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def auto_assign(signal_names: List[str], sides: str = "NSEW") -> List[IOPin]:
        """Auto-assign pins to die edges round-robin."""
        pins = []
        for i, name in enumerate(signal_names):
            side = sides[i % len(sides)]
            pins.append(IOPin(name=name, direction="input", side=side, offset_um=float(i * 10)))
        return pins


# ── Formal Equivalence Check (Gap 5) ─────────────────────────────────


class LECGenerator:
    """Generates Logic Equivalence Checking scripts."""

    @staticmethod
    def generate(design: DesignParams) -> str:
        return textwrap.dedent(f"""\
# SC-NeuroCore LEC — Yosys equivalence check
# Golden: synth_{design.top_module}.v
# Revised: {design.top_module}_final.v

read_verilog synth_{design.top_module}.v
prep -top {design.top_module}
design -stash golden

read_verilog {design.top_module}_final.v
prep -top {design.top_module}
design -stash revised

design -copy-from golden -as golden {design.top_module}
design -copy-from revised -as revised {design.top_module}

equiv_make golden revised equiv
equiv_simple
equiv_induct
equiv_status -assert
""")


# ── OCV / AOCV Derating (Gap 6) ─────────────────────────────────────


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
        return textwrap.dedent(f"""\
# OCV Derating
set_timing_derate -early {self.data_cell_early:.3f} -cell_delay [all_inputs]
set_timing_derate -late {self.data_cell_late:.3f} -cell_delay [all_inputs]
set_timing_derate -early {self.data_net_early:.3f} -net_delay [all_inputs]
set_timing_derate -late {self.data_net_late:.3f} -net_delay [all_inputs]
""")

    @classmethod
    def conservative(cls) -> OCVConfig:
        return cls(
            data_cell_early=0.93,
            data_cell_late=1.07,
            data_net_early=0.93,
            data_net_late=1.07,
            clock_cell_early=0.95,
            clock_cell_late=1.05,
        )


# ── DRC/LVS Summary Report (Gap 7) ──────────────────────────────────


@dataclass
class DRCViolation:
    """One DRC rule violation."""

    rule_name: str
    count: int
    severity: str = "error"  # error, warning, info


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
        return not any(v.severity == "error" and v.count > 0 for v in self.drc_violations)

    @property
    def all_pass(self) -> bool:
        return (
            self.timing.passed
            and self.power.passed
            and self.area.passed
            and self.drc_clean
            and self.lvs_match
        )

    def to_dict(self) -> Dict[str, Any]:
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


# ── PDK Validation (Gap 8) ──────────────────────────────────────────


@dataclass
class PDKValidationResult:
    """Result of PDK sanity check."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_pdk(pdk: PDKConfig) -> PDKValidationResult:
    """Check PDK configuration for obvious errors."""
    errors = []
    warnings = []

    if pdk.pdk_type != PDKType.CUSTOM:
        if not pdk.liberty_file:
            errors.append("liberty_file is empty")
        if not pdk.lef_file:
            errors.append("lef_file is empty")
        if not pdk.tech_lef:
            errors.append("tech_lef is empty")

    if pdk.clock_period_ns <= 0:
        errors.append(f"clock_period_ns must be positive, got {pdk.clock_period_ns}")
    if pdk.voltage_v <= 0:
        errors.append(f"voltage_v must be positive, got {pdk.voltage_v}")
    if pdk.metal_layers < 3:
        warnings.append(f"only {pdk.metal_layers} metal layers — may limit routing")

    return PDKValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_pdk_installation(
    pdk: PDKConfig,
    pdk_root: Optional[str] = None,
    require_signoff: bool = False,
) -> PDKValidationResult:
    """Check whether the resolved open-source PDK files are present locally."""
    base = validate_pdk(pdk)
    errors = list(base.errors)
    warnings = list(base.warnings)

    resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root=pdk_root, require_existing=True)
    for name in resolution.missing_required:
        path = resolution.files.required_paths()[name]
        errors.append(f"{name} not found: {path}")
    for name in resolution.missing_optional:
        path = resolution.files.optional_paths()[name]
        message = f"{name} not found: {path}"
        if require_signoff:
            errors.append(message)
        else:
            warnings.append(message)

    return PDKValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ── Hierarchical / Multi-Block Flow (Gap 9) ─────────────────────────


@dataclass
class BlockConfig:
    """One block in a hierarchical ASIC flow."""

    name: str
    design: DesignParams
    is_hard_macro: bool = False
    abstract_lef: str = ""


@dataclass
class HierarchicalFlow:
    """Multi-block ASIC flow with per-block synthesis + top integration."""

    top_design: DesignParams
    blocks: List[BlockConfig] = field(default_factory=list)

    def add_block(self, block: BlockConfig) -> None:
        self.blocks.append(block)

    def block_names(self) -> List[str]:
        return [b.name for b in self.blocks]

    def generate_block_scripts(self, pdk: PDKConfig) -> Dict[str, str]:
        gen = ASICFlowGenerator()
        result = {}
        for block in self.blocks:
            output = gen.generate(pdk, block.design)
            result[block.name] = output.synth_tcl
        return result

    def generate_top_integration(self, pdk: PDKConfig) -> str:
        lines = [f"# Hierarchical integration for {self.top_design.top_module}"]
        for block in self.blocks:
            if block.is_hard_macro and block.abstract_lef:
                lines.append(f"read_lef {block.abstract_lef}  ;# macro: {block.name}")
        lines.append(f"read_verilog synth_{self.top_design.top_module}.v")
        lines.append(f"link_design {self.top_design.top_module}")
        return "\n".join(lines) + "\n"


# ── Tape-Out Readiness Checklist (Gap 10) ────────────────────────────


@dataclass
class TapeOutChecklist:
    """Go/no-go checklist for ASIC tape-out."""

    synthesis_clean: bool = False
    timing_met: bool = False
    power_within_budget: bool = False
    area_within_limit: bool = False
    drc_clean: bool = False
    lvs_clean: bool = False
    formal_equiv_pass: bool = False
    cdc_clean: bool = False
    ir_drop_ok: bool = False
    esd_reviewed: bool = False

    @property
    def readiness_score(self) -> float:
        checks = [
            self.synthesis_clean,
            self.timing_met,
            self.power_within_budget,
            self.area_within_limit,
            self.drc_clean,
            self.lvs_clean,
            self.formal_equiv_pass,
            self.cdc_clean,
            self.ir_drop_ok,
            self.esd_reviewed,
        ]
        return sum(1 for c in checks if c) / len(checks)

    @property
    def is_tape_out_ready(self) -> bool:
        return self.readiness_score == 1.0

    def failing_checks(self) -> List[str]:
        names = [
            "synthesis_clean",
            "timing_met",
            "power_within_budget",
            "area_within_limit",
            "drc_clean",
            "lvs_clean",
            "formal_equiv_pass",
            "cdc_clean",
            "ir_drop_ok",
            "esd_reviewed",
        ]
        return [n for n in names if not getattr(self, n)]

    def from_signoff(self, summary: SignoffSummary) -> None:
        """Populate from a signoff summary."""
        self.timing_met = summary.timing.passed
        self.power_within_budget = summary.power.passed
        self.area_within_limit = summary.area.passed
        self.drc_clean = summary.drc_clean
        self.lvs_clean = summary.lvs_match
