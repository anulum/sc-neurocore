#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Generate the SC Compte ring16 RTL/Yosys/formal evidence receipt."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENV_BIN = ROOT / ".venv/bin"
RTL = ROOT / "hdl/formal/catalogue/sc_compte_wm_ring16.v"
FORMAL = ROOT / "hdl/formal/catalogue/sc_compte_wm_ring16_formal.v"
SBY = ROOT / "hdl/formal/catalogue/sc_compte_wm_ring16.sby"
SCHEMA = ROOT / "src/sc_neurocore/network/schemas/sc_compte_wm_network.json"
TEST = ROOT / "tests/test_sc_compte_wm_ring16_rtl.py"
OUTPUT = ROOT / "benchmarks/results/bench_sc_compte_wm_ring16.json"


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout + result.stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yosys_statistics(yosys: Path) -> dict[str, Any]:
    script = f"read_verilog -sv {RTL.relative_to(ROOT)}; synth -top sc_compte_wm_ring16; stat -json"
    output = _run([str(yosys), "-p", script])
    match = re.search(r'\n(\{\n\s+"creator".*)', output, flags=re.DOTALL)
    if match is None:
        raise RuntimeError("Yosys stat -json payload was not found")
    statistics, _ = json.JSONDecoder().raw_decode(match.group(1))
    module = statistics["modules"]["\\sc_compte_wm_ring16"]
    if module["num_processes"] != 0 or module["num_cells"] <= 0:
        raise RuntimeError("Yosys did not produce a structural synthesized netlist")
    return {
        "creator": statistics["creator"],
        "num_cells": module["num_cells"],
        "num_wires": module["num_wires"],
        "num_wire_bits": module["num_wire_bits"],
        "num_memories": module["num_memories"],
        "num_processes": module["num_processes"],
        "num_cells_by_type": module["num_cells_by_type"],
    }


def _run_equivalence(yosys: Path) -> None:
    rtl = RTL.relative_to(ROOT)
    script = f"""
read_verilog -sv {rtl}
hierarchy -top sc_compte_wm_ring16
proc
memory
design -stash gold
read_verilog -sv {rtl}
hierarchy -top sc_compte_wm_ring16
proc
memory
opt
design -stash gate
design -copy-from gold -as gold sc_compte_wm_ring16
design -copy-from gate -as gate sc_compte_wm_ring16
equiv_make gold gate equiv
hierarchy -top equiv
equiv_struct -icells
equiv_simple -seq 1
equiv_induct -seq 4
equiv_status -assert
"""
    _run([str(yosys), "-q", "-p", script])


def _run_formal(sby: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = f"{VENV_BIN}:{env['PATH']}"
    with tempfile.TemporaryDirectory(prefix="sc-compte-ring16-") as temporary:
        work = Path(temporary) / "proof"
        _run(
            [str(sby), "-f", "-d", str(work), SBY.name],
            cwd=SBY.parent,
            env=env,
        )
        if (work / "status").read_text(encoding="utf-8").split()[0] != "PASS":
            raise RuntimeError("SymbiYosys did not report PASS")


def main() -> None:
    """Run focused proof stages and write one source-custodied receipt."""
    yosys = VENV_BIN / "yosys"
    iverilog = VENV_BIN / "iverilog"
    vvp = VENV_BIN / "vvp"
    sby = VENV_BIN / "sby"
    cvc5 = VENV_BIN / "cvc5"
    for tool in (yosys, iverilog, vvp, sby, cvc5):
        if not tool.exists():
            raise RuntimeError(f"repository .venv is missing required tool: {tool.name}")

    pytest_output = _run(
        [
            str(VENV_BIN / "python"),
            "-m",
            "pytest",
            "-q",
            f"{TEST.relative_to(ROOT)}::test_frozen_lut_is_live_source_derived_and_schema_bounded",
            f"{TEST.relative_to(ROOT)}::test_iverilog_matches_independent_dense_oracle",
        ],
        env={**os.environ, "PYTHONPATH": "src"},
    )
    if "2 passed" not in pytest_output:
        raise RuntimeError("focused co-simulation did not report two passing tests")

    _run_formal(sby)
    synthesis = _yosys_statistics(yosys)
    _run_equivalence(yosys)

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    boundary = schema["hardware_boundary"]
    receipt = {
        "schema_version": 1,
        "identity": "SC-COMPTE-WM-NETWORK-RING16-RTL",
        "claim_boundary": {
            "enrolled": "16-bin circular E-to-E no-autapse convolution",
            "state_subset": boundary["state_subset"],
            "fixed_point_format": boundary["fixed_point_format"],
            "accumulator_format": boundary["accumulator_format"],
            "input_surface": boundary["input_surface"],
            "latency_cycles": boundary["latency_cycles"],
            "full_network_binary64_equivalence_claimed": False,
            "software_projection_error_claimed": False,
            "physical_device_evidence": False,
        },
        "quantization": {
            "weight_lut_q1616": boundary["weight_lut_q1616"],
            "weight_rule": boundary["weight_quantization"],
            "output_rule": boundary["output_quantization"],
            "integer_cosim_error_lsb": boundary["integer_cosim_error_lsb"],
        },
        "cosimulation": {
            "passed": True,
            "vectors": 4,
            "target_aggregates_checked": 29,
            "all_sixteen_targets_checked": True,
            "busy_load_rejection_checked": True,
            "oracle": "independent Python dense integer sum from live software footprint",
        },
        "formal": {
            "passed": True,
            "engine": "smtbmc cvc5",
            "depth": 20,
            "scope": [
                "synchronous reset",
                "one-cycle done pulse",
                "busy/done exclusion",
                "exact 16-cycle accepted-request latency",
            ],
            "datapath_formally_claimed": False,
        },
        "post_optimization_equivalence": {
            "passed": True,
            "gold": "Yosys proc+memory netlist",
            "revised": "same netlist after Yosys opt",
            "proof": [
                "equiv_struct -icells",
                "equiv_simple -seq 1",
                "equiv_induct -seq 4",
                "equiv_status -assert",
            ],
        },
        "synthesis": {
            "passed": True,
            "top": "sc_compte_wm_ring16",
            **synthesis,
            "device_target": None,
            "timing_claimed": False,
            "area_claimed": False,
            "power_claimed": False,
        },
        "toolchain": {
            "yosys": _run([str(yosys), "-V"]).strip(),
            "iverilog": _run([str(iverilog), "-V"]).splitlines()[0],
            "sby": _run([str(sby), "--version"]).strip(),
            "cvc5": _run([str(cvc5), "--version"]).splitlines()[0],
            "custody": "repository .venv/bin",
        },
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (RTL, FORMAL, SBY, SCHEMA, TEST, Path(__file__))
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
