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
    """Retain exact circuit observations and an explicitly requested runtime fatal.

    Attributes
    ----------
    name : str
        Corpus directory identifying the native-generated circuit.
    observations : int
        Exact ordered reference samples, including those before a requested fatal.
    mlir_sha256, verilog_sha256 : str
        Digests of the actual compiler input and exported SystemVerilog.
    expected_runtime_failure : str or None
        Fixed diagnostic proved with Icarus exit one; None requires exit zero.
    synthesis_observations : int or None
        Exact samples with SYNTHESIS defined, when separately requested.
    """

    name: str
    observations: int
    mlir_sha256: str
    verilog_sha256: str
    expected_runtime_failure: str | None = None
    synthesis_observations: int | None = None


def validate_observations(expected: Sequence[str], actual: str) -> int:
    """Require exact ordered observations, preserving unknown bits and repeated samples."""
    observations = [line for line in actual.splitlines() if line.startswith("OBS ")]
    if observations != list(expected):
        raise ValueError(
            f"reference mismatch: expected {len(expected)} observations, got {len(observations)}; "
            f"first difference: {next(((i, a, b) for i, (a, b) in enumerate(zip(expected, observations)) if a != b), 'length')}"
        )
    return len(observations)


def _run(argv: list[str], expected_failure: str | None = None) -> str:
    """Run an actual compiler/simulator with a finite deadline and actionable diagnostics."""
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    if expected_failure is not None:
        marked = any(
            line.startswith("FATAL:") and line.rstrip().endswith(": " + expected_failure)
            for line in result.stdout.splitlines()
        )
        if result.returncode != 1 or not marked:
            raise RuntimeError(
                f"expected Icarus fatal {expected_failure!r} with exit 1: {argv}\n"
                f"actual exit {result.returncode}\n{result.stdout}{result.stderr}"
            )
    elif result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {argv}\n{result.stdout}{result.stderr}"
        )
    return result.stdout


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous duplicate fields in a requested failure contract."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate runtime failure contract field")
        result[key] = value
    return result


def _expected_failure(case: Path) -> str | None:
    """Read an optional closed runtime-failure contract before invoking tools."""
    path = case / "runtime_failure.json"
    if not path.exists() and not path.is_symlink():
        return None
    raw: object = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "kind"}:
        raise ValueError("runtime failure contract requires exactly schema_version and kind")
    if raw["schema_version"] != "sc-ir-runtime-failure.v1":
        raise ValueError("unsupported runtime failure contract version")
    messages = {"overflow": "SCIR overflow", "domain": "SCIR arithmetic domain"}
    kind = raw["kind"]
    if not isinstance(kind, str) or kind not in messages:
        raise ValueError("unknown runtime failure contract kind")
    return messages[kind]


def check_circuit(case: Path, circt_opt: str = "circt-opt") -> CircuitResult:
    """Lower and execute one native circuit against its complete runtime contract.

    Parameters
    ----------
    case : Path
        Native graph.mlir, tb.sv and expected.txt inputs. An optional
        runtime_failure.json requests one closed diagnostic with Icarus exit one;
        all compiler commands still require exit zero. Without it the simulator
        must also exit zero. Optional expected_synthesis.txt separately requires
        a successful simulation with SYNTHESIS defined and its complete trace.
    circt_opt : str, optional
        Actual CIRCT executable; an unavailable producer is always an error.

    Returns
    -------
    CircuitResult
        Exact ordered observations and compiler/artifact provenance.

    Raises
    ------
    OSError
        Required inputs or tool executables are unavailable.
    ValueError
        Failure metadata or the complete observed trace differs from its contract.
    RuntimeError
        Compilation fails, an unexpected runtime error occurs or a requested
        fatal does not occur. A signal exit cannot satisfy a requested fatal.
    """
    for name in ("graph.mlir", "tb.sv", "expected.txt"):
        if not (case / name).is_file():
            raise FileNotFoundError(f"missing circuit input: {case / name}")
    source = case / "graph.mlir"
    expected_failure = _expected_failure(case)
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
    trace = _run(["vvp", str(binary)], expected_failure)
    (case / "circt-trace.txt").write_text(trace, encoding="utf-8")
    observations = validate_observations(expected, trace)
    synthesis_observations = None
    synthesis_reference = case / "expected_synthesis.txt"
    if synthesis_reference.exists() or synthesis_reference.is_symlink():
        synthesis_expected = synthesis_reference.read_text(encoding="utf-8").splitlines()
        if synthesis_expected == [""]:
            synthesis_expected = []
        synthesis_binary = case / "synthesis.vvp"
        _run(
            [
                "iverilog",
                "-g2012",
                "-DSYNTHESIS",
                "-s",
                "tb",
                "-o",
                str(synthesis_binary),
                str(sv),
                str(case / "tb.sv"),
            ]
        )
        synthesis_trace = _run(["vvp", str(synthesis_binary)])
        (case / "circt-synthesis-trace.txt").write_text(synthesis_trace, encoding="utf-8")
        synthesis_observations = validate_observations(synthesis_expected, synthesis_trace)
    return CircuitResult(
        case.name,
        observations,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        hashlib.sha256(sv.read_bytes()).hexdigest(),
        expected_failure,
        synthesis_observations,
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
