#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DM-04 committed synthesis report reader

"""DM-04: Summarise committed HDL reports (Vivado text / Yosys JSON).

Honesty
-------
Proves: parsing of **already committed** files under ``hdl/reports/``.
Does not prove: that your live board matches these numbers, or PPA in production.

Usage::

    PYTHONPATH=src python examples/dm04_synthesis_report_reader.py
    PYTHONPATH=src python examples/dm04_synthesis_report_reader.py --dir hdl/reports
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_vivado_util(text: str) -> dict[str, str]:
    """Extract a few high-signal utilization lines from Vivado util reports."""
    out: dict[str, str] = {}
    for key, pattern in (
        ("tool", r"Tool Version\s*:\s*(.+)"),
        ("date", r"Date\s*:\s*(.+)"),
        ("design", r"Design\s*:\s*(\S+)"),
        ("device", r"Device\s*:\s*(\S+)"),
    ):
        m = re.search(pattern, text)
        if m:
            out[key] = m.group(1).strip()
    # LUT / FF rows from summary table
    for label, site in (
        ("slice_luts", r"\|\s*Slice LUTs\*?\s*\|\s*(\d+)"),
        ("slice_registers", r"\|\s*Slice Registers\s*\|\s*(\d+)"),
        ("dsp", r"\|\s*DSPs\s*\|\s*(\d+)"),
        ("bram", r"\|\s*Block RAM Tile\s*\|\s*([\d.]+)"),
    ):
        m = re.search(site, text)
        if m:
            out[label] = m.group(1)
    return out


def parse_yosys_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    summary: dict[str, object] = {"file": path.name}
    if not isinstance(data, dict):
        summary["type"] = type(data).__name__
        return summary
    summary["keys"] = sorted(data.keys())[:12]
    design = data.get("design")
    if isinstance(design, dict):
        summary["design_keys"] = sorted(design.keys())[:12]
        # common yosys num_* stats if present
        for k, v in design.items():
            if isinstance(k, str) and k.startswith("num_") and isinstance(v, (int, float, str)):
                summary[k] = v
    creator = data.get("creator")
    if creator is not None:
        summary["creator"] = creator
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("hdl/reports"),
        help="Directory of committed reports",
    )
    args = parser.parse_args()
    root = args.dir
    if not root.is_dir():
        print(f"MISSING report dir: {root} — fail closed (checkout full tree).")
        return 1

    rpt_files = sorted(root.glob("*.rpt")) + sorted(root.rglob("*.rpt"))
    json_files = sorted(root.glob("*.json")) + sorted(root.rglob("yosys_*.json"))
    # de-dupe
    rpt_files = sorted({p.resolve() for p in rpt_files})
    json_files = sorted({p.resolve() for p in json_files})

    print(f"report dir: {root.resolve()}")
    print(f"found {len(rpt_files)} .rpt, {len(json_files)} yosys-like .json")

    for path in rpt_files[:12]:
        text = path.read_text(encoding="utf-8", errors="replace")
        meta = parse_vivado_util(text)
        print(f"\n== {path.name} ==")
        if not meta:
            print("  (no utilization summary lines matched)")
            continue
        for k, v in meta.items():
            print(f"  {k}: {v}")

    for path in json_files[:12]:
        print(f"\n== {path.name} ==")
        try:
            summary = parse_yosys_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  unreadable: {type(exc).__name__}")
            continue
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print(
        "\nDM-04 complete. Numbers above are from committed files only — "
        "not a live synthesis run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
