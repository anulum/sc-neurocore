# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive precision row benchmark evidence

"""Generate local, non-isolated adaptive precision row validation evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import TypedDict


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RESULT_PATH = REPO_ROOT / "benchmarks" / "results" / "bench_adaptive_precision_rows.json"

from sc_neurocore.compiler.adaptive_precision import LayerPrecision, SynapsePrecision  # noqa: E402


class CommandResult(TypedDict):
    """Serialized subprocess validation result."""

    command: list[str]
    seconds: float
    returncode: int
    status: str


class PythonTiming(TypedDict):
    """Serialized Python public-row timing result."""

    calls: int
    seconds: float
    calls_per_second: float


def _time_python(calls: int) -> dict[str, PythonTiming]:
    start = perf_counter()
    for index in range(calls):
        layer_row = LayerPrecision(
            layer_index=index % 4,
            name=f"layer_{index % 4}",
            bitstream_length=256,
            error_bound=0.03125,
            sensitivity=0.5,
        )
        if layer_row.to_dict()["bitstream_length"] != 256:
            raise RuntimeError("LayerPrecision row serialization drifted")
    layer_seconds = perf_counter() - start

    start = perf_counter()
    for index in range(calls):
        synapse_row = SynapsePrecision(
            layer_index=index % 4,
            layer_name=f"layer_{index % 4}",
            output_index=index % 8,
            input_index=(index + 1) % 8,
            bit_width=8,
            bitstream_length=128,
            sensitivity=0.5,
            quantization_error_bound=0.01,
            stochastic_error_bound=0.02,
            total_error_bound=0.03,
        )
        if synapse_row.to_dict()["total_error_bound"] != 0.03:
            raise RuntimeError("SynapsePrecision row serialization drifted")
    synapse_seconds = perf_counter() - start

    return {
        "layer_precision_rows": {
            "calls": calls,
            "seconds": round(layer_seconds, 6),
            "calls_per_second": round(calls / layer_seconds, 3),
        },
        "synapse_precision_rows": {
            "calls": calls,
            "seconds": round(synapse_seconds, 6),
            "calls_per_second": round(calls / synapse_seconds, 3),
        },
    }


def _run_command(command: list[str]) -> CommandResult:
    start = perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    seconds = perf_counter() - start
    return {
        "command": command,
        "seconds": round(seconds, 6),
        "returncode": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "fail",
    }


def main() -> int:
    """Write adaptive precision row evidence JSON and return a process status."""
    rust_binary = Path("/tmp/sc_neurocore_adaptive_precision_rows_tests")
    results: dict[str, object] = {
        "benchmark_id": "adaptive-precision-rows-polyglot-local-validation",
        "evidence_class": "local_regression_non_isolated",
        "production_benchmark_claim": False,
        "python": _time_python(calls=10_000),
        "rust_compile": _run_command(
            [
                "rustc",
                "--edition=2021",
                "--test",
                "src/sc_neurocore/accel/rust/safety/adaptive_precision.rs",
                "-o",
                str(rust_binary),
            ]
        ),
        "rust_tests": _run_command([str(rust_binary)]),
        "julia_validation": _run_command(
            [
                "julia",
                "--startup-file=no",
                "--history-file=no",
                "-e",
                (
                    'include("src/sc_neurocore/accel/julia/compiler/adaptive_precision.jl"); '
                    "using .AdaptivePrecisionAccel; "
                    'layer = LayerPrecisionState(0, "fc", 256, 0.03125, 0.5); '
                    "@assert validate_adaptive_precision(layer); "
                    'try LayerPrecisionState(-1, "fc", 256, 0.03125, 0.5); error("layer_index accepted"); '
                    "catch err; @assert err isa ArgumentError; end; "
                    'syn = SynapsePrecisionState(0, "fc", 1, 2, 8, 128, 0.5, 0.01, 0.02, 0.03); '
                    "@assert validate_synapse_precision(syn); "
                    'try SynapsePrecisionState(-1, "fc", 1, 2, 8, 128, 0.5, 0.01, 0.02, 0.03); error("layer_index accepted"); '
                    "catch err; @assert err isa ArgumentError; end; "
                    '@assert to_dict(syn)["total_error_bound"] == 0.03'
                ),
            ]
        ),
        "mojo_validation": _run_command(
            ["mojo", "src/sc_neurocore/accel/mojo/kernels/adaptive_precision.mojo"]
        ),
    }

    failed = [
        name
        for name, value in results.items()
        if isinstance(value, dict) and value.get("status") == "fail"
    ]
    results["status"] = "fail" if failed else "pass"
    results["failed_checks"] = failed

    RESULT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
