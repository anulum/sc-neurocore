# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parse Vivado implementation reports into structured JSON

"""
Parse Vivado implementation reports into structured JSON.

Usage::

    python tools/vivado_report.py vivado_reports/
    python tools/vivado_report.py vivado_reports/ --json benchmarks/results/vivado_impl.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_timing(path: Path) -> dict:
    """Extract WNS/TNS and compute Fmax from timing_summary.rpt."""
    text = path.read_text(errors="replace")
    result = {"wns_ns": None, "tns_ns": None, "fmax_mhz": None, "period_ns": None}

    m = re.search(r"Design Timing Summary[\s\S]*?WNS\(ns\)\s+TNS\(ns\)", text)
    if not m:
        return result

    # WNS/TNS line is typically two lines after the header
    lines = text[m.end() :].strip().splitlines()
    for line in lines[:5]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                result["wns_ns"] = float(parts[0])
                result["tns_ns"] = float(parts[1])
                break
            except ValueError:
                continue

    # Find clock period
    m_clk = re.search(r"create_clock\s+-period\s+([\d.]+)", text)
    if not m_clk:
        m_clk = re.search(r"Waveform\s*\{[\d. ]+\}\s+([\d.]+)", text)
    if m_clk:
        result["period_ns"] = float(m_clk.group(1))

    if result["wns_ns"] is not None and result["period_ns"]:
        result["fmax_mhz"] = round(1000.0 / (result["period_ns"] - result["wns_ns"]), 1)

    return result


def parse_utilization(path: Path) -> dict:
    """Extract LUT/FF/BRAM/DSP counts from utilization report."""
    text = path.read_text(errors="replace")
    result = {"luts": None, "ffs": None, "bram_36k": None, "dsp48": None}

    for line in text.splitlines():
        if "Slice LUTs" in line or "CLB LUTs" in line:
            m = re.search(r"\|\s*(\d+)\s*\|", line)
            if m:
                result["luts"] = int(m.group(1))
        elif "Slice Registers" in line or "CLB Registers" in line:
            m = re.search(r"\|\s*(\d+)\s*\|", line)
            if m:
                result["ffs"] = int(m.group(1))
        elif "Block RAM Tile" in line:
            m = re.search(r"\|\s*([\d.]+)\s*\|", line)
            if m:
                result["bram_36k"] = float(m.group(1))
        elif "DSPs" in line or "DSP48" in line:
            m = re.search(r"\|\s*(\d+)\s*\|", line)
            if m:
                result["dsp48"] = int(m.group(1))

    return result


def parse_power(path: Path) -> dict:
    """Extract static/dynamic power from power report."""
    text = path.read_text(errors="replace")
    result = {"total_w": None, "dynamic_w": None, "static_w": None}

    for line in text.splitlines():
        if "Total On-Chip Power" in line:
            m = re.search(r"([\d.]+)\s*W", line)
            if m:
                result["total_w"] = float(m.group(1))
        elif "Dynamic" in line and "W" in line and result["dynamic_w"] is None:
            m = re.search(r"([\d.]+)\s*W", line)
            if m:
                result["dynamic_w"] = float(m.group(1))
        elif "Device Static" in line:
            m = re.search(r"([\d.]+)\s*W", line)
            if m:
                result["static_w"] = float(m.group(1))

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse Vivado implementation reports")
    ap.add_argument("report_dir", type=Path, help="directory with Vivado .rpt files")
    ap.add_argument("--json", type=str, help="write parsed results to JSON")
    args = ap.parse_args()

    d = args.report_dir
    if not d.is_dir():
        print(f"ERROR: {d} is not a directory")
        return 1

    report = {}

    timing_rpt = d / "timing_summary.rpt"
    if timing_rpt.exists():
        report["timing"] = parse_timing(timing_rpt)

    util_rpt = d / "utilization_impl.rpt"
    if not util_rpt.exists():
        util_rpt = d / "utilization_synth.rpt"
    if util_rpt.exists():
        report["utilization"] = parse_utilization(util_rpt)

    power_rpt = d / "power.rpt"
    if power_rpt.exists():
        report["power"] = parse_power(power_rpt)

    # Print summary
    t = report.get("timing", {})
    u = report.get("utilization", {})
    p = report.get("power", {})

    print("=== Vivado Implementation Results ===")
    if t.get("fmax_mhz"):
        print(f"  Fmax:    {t['fmax_mhz']} MHz (WNS: {t['wns_ns']} ns)")
    if u.get("luts"):
        print(f"  LUTs:    {u['luts']}")
        print(f"  FFs:     {u['ffs']}")
        print(f"  BRAM:    {u.get('bram_36k', 0)}")
        print(f"  DSP48:   {u.get('dsp48', 0)}")
    if p.get("total_w"):
        print(f"  Power:   {p['total_w']} W (dynamic: {p['dynamic_w']} W)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nWritten to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
