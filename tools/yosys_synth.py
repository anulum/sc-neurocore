# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Yosys FPGA Synthesis Runner

"""
Yosys FPGA Synthesis Runner
============================

Invokes Yosys on each SC-NeuroCore HDL module and parses LUT/FF/BRAM counts.

When ``sv2v`` is on PATH, SystemVerilog sources (unpacked-array ports) are
preprocessed to plain Verilog first.  Without sv2v the five SV-dependent
modules are skipped.

Usage::

    python tools/yosys_synth.py                     # all modules
    python tools/yosys_synth.py --module sc_lif_neuron
    python tools/yosys_synth.py --json benchmarks/results/yosys_synth.json
    python tools/yosys_synth.py --allow-skips       # CI artifact mode
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

MODULES = [
    "sc_bitstream_encoder",
    "sc_lif_neuron",
    "sc_bitstream_synapse",
    "sc_dotproduct_to_current",
    "sc_firing_rate_bank",
    "sc_axil_cfg",
    "sc_dense_layer_core",
    "sc_dense_layer_top",
    "sc_dense_matrix_layer",
    "sc_neurocore_top",
]

# Yosys cannot parse SV unpacked-array ports (issue #2717, open since 2021).
# These modules are skipped when sv2v is not available.
_SV_MODULES = frozenset(
    {
        "sc_axil_cfg",
        "sc_firing_rate_bank",
        "sc_dense_layer_core",
        "sc_dense_layer_top",
        "sc_neurocore_top",
    }
)

# SHD-specific modules with $readmemh initialisation from external weight /
# delay hex files. These are not part of the general SC-NeuroCore core and
# are synthesised through a dedicated Vivado flow that stages the hex files
# alongside the bitstream build; including them in the general yosys
# resource counter fails at parse time because the hex files are not
# available in the repository root.
_SHD_MODULES = frozenset(
    {
        "sc_vmin_lif_neuron",
        "sc_axonal_delay",
        "sc_dense_int8_sparse",
        "sc_shd_top",
    }
)

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
    """Extract resource counts from the LAST Yosys 'stat' block.

    Yosys prints intermediate stats during synth_xilinx; only the final
    block (from the explicit ``stat`` command) is authoritative.
    Format: ``  <count>   <CELL_TYPE>``  (count before name).
    """
    blocks = text.split("Printing statistics.")
    if len(blocks) < 2:
        return {"luts": 0, "ffs": 0, "bram": 0, "dsp": 0}
    last_block = blocks[-1]

    counts: dict[str, int] = {"luts": 0, "ffs": 0, "bram": 0, "dsp": 0}
    for line in last_block.splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)", line)
        if not m:
            continue
        n, cell = int(m.group(1)), m.group(2)
        if cell in ("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"):
            counts["luts"] += n
        elif cell in ("FDRE", "FDSE", "FDCE", "FDPE"):
            counts["ffs"] += n
        elif cell.startswith("RAMB"):
            counts["bram"] += n
        elif cell.startswith("DSP48"):
            counts["dsp"] += n

    return counts


def preprocess_hdl() -> list[Path]:
    """Collect HDL sources. Preprocesses through sv2v when available.

    Testbenches (`tb_*.v`) and SHD-specific modules (see `_SHD_MODULES`)
    are excluded — the SHD modules use `$readmemh` to load external
    weight / delay hex files that are staged into cosim temp directories,
    not into the repo root where this tool runs.
    """
    hdl_dir = REPO_ROOT / "hdl"
    all_v = sorted(
        f
        for f in hdl_dir.glob("*.v")
        if not f.name.startswith("tb_") and f.stem not in _SHD_MODULES
    )

    sv2v = shutil.which("sv2v")
    if sv2v:
        converted = Path(tempfile.mkdtemp()) / "sc_neurocore_converted.v"
        result = subprocess.run(
            [sv2v, *[str(f) for f in all_v], "-w", str(converted)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  sv2v: preprocessed {len(all_v)} files -> {converted.name}")
            return [converted]
        print(f"  sv2v failed ({result.stderr[:200]}), falling back to filtered sources")

    return [f for f in all_v if f.stem not in _SV_MODULES]


def _build_yosys_commands(module: str, sources: list[Path]) -> str:
    """Generate inline Yosys commands.

    `-DSYNTHESIS` disables simulation-only `$readmemh` initial blocks
    (e.g. in `sc_dense_int8_sparse.v`) that reference weight hex files
    that only exist inside cosim temp directories. Real FPGA builds
    write the weight ROM through AXI at boot time, so skipping the
    init during resource-count synthesis does not affect results.
    """
    cmds = [f"read_verilog -DSYNTHESIS {f}" for f in sources]
    cmds.append(f"synth_xilinx -top {module} -flatten")
    cmds.append("stat")
    return "; ".join(cmds)


def run_synth(module: str, sources: list[Path]) -> SynthResult:
    yosys_cmds = _build_yosys_commands(module, sources)
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
    ap.add_argument(
        "--allow-skips",
        action="store_true",
        help="return success when skipped modules are recorded as artifact evidence",
    )
    args = ap.parse_args()

    if not shutil.which("yosys"):
        print("WARNING: yosys not found in PATH")
        print("Install: https://github.com/YosysHQ/yosys")
        return 1

    sources = preprocess_hdl()
    has_sv2v = shutil.which("sv2v") is not None
    modules = [args.module] if args.module else MODULES

    results: list[SynthResult] = []
    for mod in modules:
        if not has_sv2v and mod in _SV_MODULES:
            print(f"  Synthesizing {mod}... SKIP (needs sv2v)")
            results.append(SynthResult(mod, 0, 0, 0, 0, False, "needs sv2v for SV features"))
            continue
        print(f"  Synthesizing {mod}...", end=" ", flush=True)
        r = run_synth(mod, sources)
        if r.ok:
            print(f"{r.luts} LUTs, {r.ffs} FFs")
        else:
            print(f"SKIP ({r.error})")
        results.append(r)

    if args.markdown:
        print("\n" + format_markdown(results))

    if args.json:
        yosys_ver = subprocess.run(["yosys", "-V"], capture_output=True, text=True).stdout.strip()
        sv2v_ver = ""
        if has_sv2v:
            sv2v_ver = subprocess.run(
                ["sv2v", "--version"], capture_output=True, text=True
            ).stdout.strip()
        payload = {
            "tool": "yosys",
            "yosys_version": yosys_ver,
            "sv2v_version": sv2v_ver,
            "target": "Xilinx 7-series (synth_xilinx -flatten)",
            "modules": [asdict(r) for r in results],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nResults written to {args.json}")

    return 0 if args.allow_skips or all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
