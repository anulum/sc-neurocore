# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Few-shot HAAM benchmark evidence

"""Generate local, non-isolated HAAM validation and timing evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = REPO_ROOT / "benchmarks" / "results" / "bench_few_shot_haam.json"


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


@dataclass(frozen=True)
class EpisodeFixture:
    """Deterministic support/query fixture for HAAM timing."""

    support_x: list[NDArray[np.float64]]
    support_y: list[int]
    query_x: list[NDArray[np.float64]]


def _fixture() -> EpisodeFixture:
    rng = np.random.default_rng(42)
    support_x: list[NDArray[np.float64]] = []
    support_y: list[int] = []
    for label in range(5):
        pattern = np.zeros(64, dtype=np.float64)
        pattern[label * 8 : label * 8 + 8] = rng.random(8)
        support_x.append(pattern)
        support_y.append(label)

    query_x: list[NDArray[np.float64]] = []
    for label in range(5):
        query = np.zeros(64, dtype=np.float64)
        query[label * 8 : label * 8 + 8] = rng.random(8)
        query_x.append(query)

    return EpisodeFixture(support_x=support_x, support_y=support_y, query_x=query_x)


def _time_python(calls: int) -> dict[str, PythonTiming]:
    from sc_neurocore.few_shot import HebbianFewShot, SpikePrototypeNet

    fixture = _fixture()
    learner = HebbianFewShot(n_features=64, n_classes=5, lr_hebbian=0.1)
    proto = SpikePrototypeNet(n_features=64, metric="cosine")

    start = perf_counter()
    for _ in range(calls):
        predictions = learner.few_shot_episode(
            fixture.support_x,
            fixture.support_y,
            fixture.query_x,
        )
        if predictions != [0, 1, 2, 3, 4]:
            raise RuntimeError("HebbianFewShot predictions drifted")
    hebbian_seconds = perf_counter() - start

    start = perf_counter()
    for _ in range(calls):
        predictions = proto.classify(fixture.support_x, fixture.support_y, fixture.query_x)
        if predictions != [0, 1, 2, 3, 4]:
            raise RuntimeError("SpikePrototypeNet predictions drifted")
    proto_seconds = perf_counter() - start

    return {
        "hebbian_episode": {
            "calls": calls,
            "seconds": round(hebbian_seconds, 6),
            "calls_per_second": round(calls / hebbian_seconds, 3),
        },
        "prototype_classify": {
            "calls": calls,
            "seconds": round(proto_seconds, 6),
            "calls_per_second": round(calls / proto_seconds, 3),
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
    """Write HAAM local evidence JSON and return a process status."""
    rust_binary = Path("/tmp/sc_neurocore_haam_rust_tests")
    results: dict[str, object] = {
        "benchmark_id": "few-shot-haam-polyglot-local-validation",
        "evidence_class": "local_regression_non_isolated",
        "production_benchmark_claim": False,
        "python": _time_python(calls=1000),
        "rust_compile": _run_command(
            [
                "rustc",
                "--edition=2021",
                "--test",
                "src/sc_neurocore/accel/rust/safety/haam.rs",
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
                    'include("src/sc_neurocore/accel/julia/few_shot/haam.jl"); '
                    "using .HaamAccel; "
                    "@assert validate_haam(); "
                    'net = SpikePrototypeNetState(3, "hamming"); '
                    "@assert classify!(net, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], "
                    "[0, 1], [[0.8, 0.1, 0.0]]) == [0]"
                ),
            ]
        ),
        "mojo_validation": _run_command(["mojo", "src/sc_neurocore/accel/mojo/kernels/haam.mojo"]),
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
