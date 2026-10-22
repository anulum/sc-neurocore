# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Numerical CIRCT checks for public native IR circuits

"""Lower native Rust test circuits with CIRCT and compare complete reference observations.

Set ``SCN_IR_CIRCT_RECEIPT_DIR`` when running the native streaming integration
tests, then run this tool on that directory. No missing tools, empty corpora or
failed compiler/simulator commands are accepted as successful validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class CircuitResult:
    """One fully lowered and simulated circuit with its exact input/output provenance."""

    name: str
    observations: int
    mlir_sha256: str
    verilog_sha256: str


def validate_observations(expected: Sequence[str], actual: str) -> int:
    """Require exact ordered observations, preserving unknown bits and repeated samples."""
    observations = [line for line in actual.splitlines() if line.startswith("OBS ")]
    if observations != list(expected):
        raise ValueError(
            f"reference mismatch: expected {len(expected)} observations, got {len(observations)}; "
            f"first difference: {next(((i, a, b) for i, (a, b) in enumerate(zip(expected, observations)) if a != b), 'length')}"
        )
    return len(observations)


def _run(argv: list[str]) -> str:
    """Run an actual compiler/simulator with a finite deadline and actionable diagnostics."""
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {argv}\n{result.stdout}{result.stderr}"
        )
    return result.stdout


def check_circuit(case: Path, circt_opt: str = "circt-opt") -> CircuitResult:
    """Check one native-generated module through real CIRCT, Icarus and its golden trace."""
    for name in ("graph.mlir", "tb.sv", "expected.txt"):
        if not (case / name).is_file():
            raise FileNotFoundError(f"missing circuit input: {case / name}")
    source = case / "graph.mlir"
    expected = (case / "expected.txt").read_text(encoding="utf-8").splitlines()
    # An empty output tuple has no observations; the writer emits a trailing newline.
    if expected == [""]:
        expected = []
    verilog = _run(
        [
            circt_opt,
            str(source),
            "--lower-seq-to-sv=disable-reg-randomization",
            "--canonicalize",
            "--export-verilog",
            "-o",
            str(case / "lowered.mlir"),
        ]
    )
    # CIRCT emits Verilog to stdout; -o receives the transformed MLIR.
    sv = case / "lowered.sv"
    sv.write_text(verilog, encoding="utf-8")
    binary = case / "lowered.vvp"
    _run(["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(sv), str(case / "tb.sv")])
    trace = _run(["vvp", str(binary)])
    (case / "circt-trace.txt").write_text(trace, encoding="utf-8")
    observations = validate_observations(expected, trace)
    return CircuitResult(
        case.name,
        observations,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        hashlib.sha256(sv.read_bytes()).hexdigest(),
    )


def check_cases(root: Path, circt_opt: str = "circt-opt") -> list[CircuitResult]:
    """Check every case, refusing a missing or empty native test corpus."""
    cases = sorted(path for path in root.iterdir() if path.is_dir())
    if not cases:
        raise ValueError(f"no CIRCT cases in {root}; run the native integration tests first")
    return [check_circuit(case, circt_opt) for case in cases]


def main() -> int:
    """Run the numerical gate and optionally save a machine-readable result receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--circt-opt", default="circt-opt")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        results = check_cases(args.cases, args.circt_opt)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.report is not None:
        args.report.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Checked {len(results)} circuits, {sum(result.observations for result in results)} exact observations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
