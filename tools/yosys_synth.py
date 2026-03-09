# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Yosys FPGA Synthesis Runner
============================

Invokes Yosys on each SC-NeuroCore HDL module and parses LUT/FF/BRAM counts.

Usage::

    python tools/yosys_synth.py                     # all modules
    python tools/yosys_synth.py --module sc_lif_neuron
    python tools/yosys_synth.py --json benchmarks/results/yosys_synth.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

MODULES = [
    "sc_bitstream_encoder",
    "sc_lif_neuron",
    "sc_bitstream_synapse",
    "sc_dotproduct_to_current",
    "sc_firing_rate_bank",
    "sc_dense_layer_core",
    "sc_dense_layer_top",
    "sc_neurocore_top",
]

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SynthResult:
    module: str
    luts: int
    ffs: int
    bram: int
    dsp: int
    ok: bool
    error: str = ""


def parse_stat_output(text: str) -> dict[str, int]:
    """Extract resource counts from Yosys 'stat' output."""
    counts: dict[str, int] = {"luts": 0, "ffs": 0, "bram": 0, "dsp": 0}

    for line in text.splitlines():
        line = line.strip()
        # LUT counts: various LUT types in Xilinx
        for lut_type in ("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"):
            m = re.match(rf"{lut_type}\s+(\d+)", line)
            if m:
                counts["luts"] += int(m.group(1))
        # Flip-flops
        for ff_type in ("FDRE", "FDSE", "FDCE", "FDPE"):
            m = re.match(rf"{ff_type}\s+(\d+)", line)
            if m:
                counts["ffs"] += int(m.group(1))
        # BRAM
        if re.match(r"RAMB\d+", line):
            m = re.match(r"RAMB\w+\s+(\d+)", line)
            if m:
                counts["bram"] += int(m.group(1))
        # DSP
        m = re.match(r"DSP48E\d?\s+(\d+)", line)
        if m:
            counts["dsp"] += int(m.group(1))

    return counts


def _build_yosys_commands(module: str) -> str:
    """Generate inline Yosys commands (avoids TCL dependency)."""
    hdl_dir = REPO_ROOT / "hdl"
    cmds = []
    for f in sorted(hdl_dir.glob("*.v")):
        if f.name.startswith("tb_"):
            continue
        cmds.append(f"read_verilog -sv {f}")
    cmds.append(f"synth_xilinx -top {module} -flatten")
    cmds.append("stat")
    return "; ".join(cmds)


def run_synth(module: str) -> SynthResult:
    yosys_cmds = _build_yosys_commands(module)
    try:
        result = subprocess.run(
            ["yosys", "-p", yosys_cmds],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
            check=True,
        )
    except FileNotFoundError:
        return SynthResult(module, 0, 0, 0, 0, False, "yosys not found in PATH")
    except subprocess.TimeoutExpired:
        return SynthResult(module, 0, 0, 0, 0, False, "synthesis timed out (120s)")
    except subprocess.CalledProcessError as e:
        err = e.stderr[:500] if e.stderr else "unknown error"
        return SynthResult(module, 0, 0, 0, 0, False, err)

    counts = parse_stat_output(result.stdout)
    return SynthResult(
        module=module,
        luts=counts["luts"],
        ffs=counts["ffs"],
        bram=counts["bram"],
        dsp=counts["dsp"],
        ok=True,
    )


def format_markdown(results: list[SynthResult]) -> str:
    lines = [
        "| Module | LUTs | FFs | BRAM | DSP | Status |",
        "|--------|-----:|----:|-----:|----:|--------|",
    ]
    for r in results:
        status = "OK" if r.ok else f"SKIP: {r.error}"
        lines.append(f"| `{r.module}` | {r.luts} | {r.ffs} | {r.bram} | {r.dsp} | {status} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Yosys FPGA synthesis for SC-NeuroCore HDL")
    ap.add_argument("--module", type=str, help="single module to synthesize")
    ap.add_argument("--json", type=str, help="write results to JSON file")
    ap.add_argument("--markdown", action="store_true", help="print markdown table")
    args = ap.parse_args()

    modules = [args.module] if args.module else MODULES

    if not shutil.which("yosys"):
        print("WARNING: yosys not found in PATH")
        print("Install: https://github.com/YosysHQ/yosys or 'pip install yosys'")
        print("Generating placeholder results...")

    results: list[SynthResult] = []
    for mod in modules:
        print(f"  Synthesizing {mod}...", end=" ", flush=True)
        r = run_synth(mod)
        if r.ok:
            print(f"{r.luts} LUTs, {r.ffs} FFs")
        else:
            print(f"SKIP ({r.error})")
        results.append(r)

    if args.markdown:
        print("\n" + format_markdown(results))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nResults written to {args.json}")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
