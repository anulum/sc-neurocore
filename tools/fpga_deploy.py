# SPDX-License-Identifier: AGPL-3.0-or-later
"""
FPGA Deployment Pipeline
=========================

End-to-end flow: Python network description -> Rust IR -> Verilog RTL
-> Vivado/Quartus synthesis.

Usage::

    # Generate Verilog from a network description (no FPGA tools needed)
    python tools/fpga_deploy.py --emit-verilog --out build/rtl/

    # Full synthesis (requires Vivado in PATH)
    python tools/fpga_deploy.py --synth vivado --part xc7a35t --out build/vivado/

    # Full synthesis (requires Quartus in PATH)
    python tools/fpga_deploy.py --synth quartus --part 10CL025YU256C8G --out build/quartus/

    # Utilisation report only
    python tools/fpga_deploy.py --synth vivado --part xc7a35t --report-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HDL_DIR = Path(__file__).resolve().parent.parent / "hdl"
DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy" / "fpga"

SUPPORTED_PARTS = {
    "vivado": {
        "xc7a35t": "Artix-7 35T (Arty A7-35T, Basys 3)",
        "xc7a100t": "Artix-7 100T (Arty A7-100T, Nexys A7)",
        "xc7z020": "Zynq-7020 (PYNQ-Z2, ZedBoard)",
        "xczu3eg": "Zynq UltraScale+ ZU3EG (Ultra96)",
    },
    "quartus": {
        "10CL025YU256C8G": "Cyclone 10 LP 25K LE (DE10-Lite)",
        "5CSEMA5F31C6": "Cyclone V SE (DE1-SoC)",
    },
}

VERILOG_SOURCES = [
    "sc_bitstream_encoder.v",
    "sc_bitstream_synapse.v",
    "sc_dotproduct_to_current.v",
    "sc_lif_neuron.v",
    "sc_firing_rate_bank.v",
    "sc_dense_layer_core.v",
    "sc_dense_layer_top.v",
    "sc_neurocore_top.v",
    "sc_axil_cfg.v",
]


def emit_verilog(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in VERILOG_SOURCES:
        src = HDL_DIR / name
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        dst = out_dir / name
        shutil.copy2(src, dst)
        copied.append(dst)
        print(f"  {name} -> {dst}")
    return copied


def generate_vivado_tcl(part: str, out_dir: Path, sources: list[Path]) -> Path:
    tcl_path = out_dir / "synth.tcl"
    constraints = DEPLOY_DIR / "constraints.xdc"

    lines = [
        f'create_project sc_neurocore_fpga {out_dir / "project"} -part {part} -force',
        "",
    ]
    for s in sources:
        lines.append(f"add_files {s}")
    lines.append("set_property top sc_neurocore_top [current_fileset]")

    if constraints.exists():
        lines.append(f"add_files -fileset constrs_1 {constraints}")

    lines += [
        "",
        "launch_runs synth_1 -jobs 4",
        "wait_on_run synth_1",
        "",
        "open_run synth_1",
        f'report_utilization -file {out_dir / "utilization.rpt"}',
        f'report_timing_summary -file {out_dir / "timing.rpt"}',
        f'report_power -file {out_dir / "power.rpt"}',
        "",
        "launch_runs impl_1 -to_step write_bitstream -jobs 4",
        "wait_on_run impl_1",
        "",
        f'puts "Synthesis complete. Bitstream: {out_dir}/project/sc_neurocore_fpga.runs/impl_1/*.bit"',
    ]

    tcl_path.write_text("\n".join(lines) + "\n")
    return tcl_path


def generate_quartus_tcl(part: str, out_dir: Path, sources: list[Path]) -> Path:
    tcl_path = out_dir / "synth.tcl"

    lines = [
        "load_package flow",
        f"project_new sc_neurocore_fpga -overwrite -part {part}",
        "",
    ]
    for s in sources:
        lines.append(f'set_global_assignment -name VERILOG_FILE "{s}"')
    lines.append("set_global_assignment -name TOP_LEVEL_ENTITY sc_neurocore_top")

    lines += [
        "",
        "execute_flow -compile",
        "",
        'puts "Synthesis complete."',
    ]

    tcl_path.write_text("\n".join(lines) + "\n")
    return tcl_path


def run_vivado(tcl_path: Path) -> int:
    vivado = shutil.which("vivado")
    if not vivado:
        print("ERROR: vivado not found in PATH")
        print("Install Vivado from https://www.xilinx.com/support/download.html")
        return 1
    cmd = [vivado, "-mode", "batch", "-source", str(tcl_path)]
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_quartus(tcl_path: Path) -> int:
    quartus = shutil.which("quartus_sh")
    if not quartus:
        print("ERROR: quartus_sh not found in PATH")
        print(
            "Install Quartus from https://www.intel.com/content/www/us/en/products/details/fpga/development-tools/quartus-prime.html"
        )
        return 1
    cmd = [quartus, "-t", str(tcl_path)]
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="SC-NeuroCore FPGA deployment pipeline")
    ap.add_argument("--emit-verilog", action="store_true", help="copy Verilog sources only")
    ap.add_argument("--synth", choices=["vivado", "quartus"], help="run synthesis")
    ap.add_argument("--part", type=str, default="xc7a35t", help="FPGA part number")
    ap.add_argument("--out", type=str, default="build/fpga", help="output directory")
    ap.add_argument(
        "--report-only", action="store_true", help="synthesis + reports, skip bitstream"
    )
    ap.add_argument("--list-parts", action="store_true", help="list supported FPGA parts")
    args = ap.parse_args()

    if args.list_parts:
        for tool, parts in SUPPORTED_PARTS.items():
            print(f"\n{tool}:")
            for part, desc in parts.items():
                print(f"  {part:25s} {desc}")
        return

    out = Path(args.out)

    print("SC-NeuroCore FPGA Pipeline")
    print("=" * 40)

    print("\n[1/3] Emitting Verilog RTL")
    rtl_dir = out / "rtl"
    sources = emit_verilog(rtl_dir)
    if not sources:
        print("ERROR: no Verilog sources found")
        sys.exit(1)

    if args.emit_verilog and not args.synth:
        print(f"\nVerilog emitted to {rtl_dir}/")
        print("Run with --synth vivado|quartus to synthesise.")
        return

    if args.synth == "vivado":
        print(f"\n[2/3] Generating Vivado Tcl for {args.part}")
        tcl = generate_vivado_tcl(args.part, out, sources)
        print(f"  Tcl script: {tcl}")
        print("\n[3/3] Running Vivado synthesis")
        rc = run_vivado(tcl)
    elif args.synth == "quartus":
        print(f"\n[2/3] Generating Quartus Tcl for {args.part}")
        tcl = generate_quartus_tcl(args.part, out, sources)
        print(f"  Tcl script: {tcl}")
        print("\n[3/3] Running Quartus synthesis")
        rc = run_quartus(tcl)
    else:
        print("\nSpecify --synth vivado|quartus to run synthesis.")
        return

    sys.exit(rc)


if __name__ == "__main__":
    main()
