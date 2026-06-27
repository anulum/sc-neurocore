# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contrastive SSL benchmark evidence

"""Generate local, non-isolated contrastive SSL validation evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RESULT_PATH = REPO_ROOT / "benchmarks" / "results" / "bench_contrastive_ssl.json"


class CommandResult(TypedDict):
    """Serialized subprocess validation result."""

    command: list[str]
    seconds: float
    returncode: int
    status: str


class PythonTiming(TypedDict):
    """Serialized Python public-API timing result."""

    calls: int
    seconds: float
    calls_per_second: float


def _fixture() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    view_a = rng.normal(size=(32, 64)).astype(np.float64)
    view_b = view_a + rng.normal(scale=0.05, size=(32, 64)).astype(np.float64)
    return view_a, view_b


def _time_python(calls: int) -> dict[str, PythonTiming]:
    from sc_neurocore.contrastive import CSDPRule, SpikeContrastiveLoss

    view_a, view_b = _fixture()
    loss_fn = SpikeContrastiveLoss(temperature=0.5)
    rule = CSDPRule(lr=0.1, decay=0.01)
    weights = np.array([[0.2, 0.4], [0.1, 0.3]], dtype=np.float64)
    pos_pre = np.array([1.0, 0.5], dtype=np.float64)
    pos_post = np.array([0.25, 1.0], dtype=np.float64)
    neg_pre = np.array([0.0, 1.0], dtype=np.float64)
    neg_post = np.array([0.5, 0.5], dtype=np.float64)

    start = perf_counter()
    for _ in range(calls):
        value = loss_fn.compute(view_a, view_b)
        if not np.isfinite(value):
            raise RuntimeError("SpikeContrastiveLoss returned a non-finite loss")
    loss_seconds = perf_counter() - start

    start = perf_counter()
    for _ in range(calls):
        updated = rule.contrastive_step(weights, pos_pre, pos_post, neg_pre, neg_post)
        if updated.shape != weights.shape:
            raise RuntimeError("CSDPRule returned the wrong weight shape")
    csdp_seconds = perf_counter() - start

    return {
        "infonce_compute": {
            "calls": calls,
            "seconds": round(loss_seconds, 6),
            "calls_per_second": round(calls / loss_seconds, 3),
        },
        "csdp_contrastive_step": {
            "calls": calls,
            "seconds": round(csdp_seconds, 6),
            "calls_per_second": round(calls / csdp_seconds, 3),
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
    """Write contrastive SSL local evidence JSON and return a process status."""
    rust_binary = Path("/tmp/sc_neurocore_ssl_rust_tests")
    results: dict[str, object] = {
        "benchmark_id": "contrastive-ssl-polyglot-local-validation",
        "evidence_class": "local_regression_non_isolated",
        "production_benchmark_claim": False,
        "python": _time_python(calls=1000),
        "rust_compile": _run_command(
            [
                "rustc",
                "--edition=2021",
                "--test",
                "src/sc_neurocore/accel/rust/safety/ssl.rs",
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
                    'include("src/sc_neurocore/accel/julia/contrastive/ssl.jl"); '
                    "using .SslAccel; using LinearAlgebra; "
                    "@assert validate_ssl(); "
                    "loss = SpikeContrastiveLossState(0.5); "
                    "view = Matrix{Float64}(I, 4, 4); "
                    "@assert compute(loss, view, view) < compute(loss, view, view[[2,3,4,1], :])"
                ),
            ]
        ),
        "mojo_validation": _run_command(["mojo", "src/sc_neurocore/accel/mojo/kernels/ssl.mojo"]),
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
