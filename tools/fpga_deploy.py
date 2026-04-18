#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""FPGA deployment helper — list supported parts and emit Verilog sources."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PARTS = {
    "xc7a35t": "Xilinx Artix-7 35T (Basys3 / Arty A7-35)",
    "xc7a100t": "Xilinx Artix-7 100T (Arty A7-100 / Nexys A7)",
    "xc7z020": "Xilinx Zynq-7020 (PYNQ-Z2)",
    "xc7k325t": "Xilinx Kintex-7 325T",
    "Cyclone_V_5CSEMA5": "Intel Cyclone V (DE1-SoC)",
    "Cyclone_10LP_10CL025": "Intel Cyclone 10 LP",
    "ECP5_LFE5U_25F": "Lattice ECP5 25F (ULX3S)",
    "ICE40UP5K": "Lattice iCE40 UltraPlus 5K (iCEBreaker)",
}

HDL_DIR = Path(__file__).resolve().parent.parent / "hdl"


def list_parts() -> None:
    for part, desc in PARTS.items():
        print(f"  {part:30s} {desc}")


def emit_verilog(out: Path) -> None:
    rtl_dir = out / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    for v in HDL_DIR.glob("*.v"):
        if v.name.startswith("tb_"):
            continue
        shutil.copy2(v, rtl_dir / v.name)
    print(f"Copied RTL sources to {rtl_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FPGA deployment helper")
    ap.add_argument("--list-parts", action="store_true", help="Show supported FPGA parts")
    ap.add_argument("--emit-verilog", action="store_true", help="Copy synthesisable HDL")
    ap.add_argument("--out", type=Path, default=Path("build"), help="Output directory")
    args = ap.parse_args()

    if args.list_parts:
        list_parts()
    if args.emit_verilog:
        emit_verilog(args.out)
    if not args.list_parts and not args.emit_verilog:
        ap.print_help()


if __name__ == "__main__":
    main()
