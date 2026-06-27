# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Transfer checkpoint benchmark evidence

"""Generate local, non-isolated transfer checkpoint validation evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TypedDict

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RESULT_PATH = REPO_ROOT / "benchmarks" / "results" / "bench_transfer.json"

from sc_neurocore.transfer import SNNCheckpoint  # noqa: E402


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


def _checkpoint_fixture() -> SNNCheckpoint:
    rng = np.random.default_rng(42)
    return SNNCheckpoint(
        weights=[
            rng.normal(size=(32, 64)).astype(np.float64),
            rng.normal(size=(10, 32)).astype(np.float64),
        ],
        layer_names=["hidden", "output"],
        layer_sizes=[(64, 32), (32, 10)],
        neuron_types=["LIF", "LIF"],
        metadata={"task": "transfer-bench", "accuracy": 0.95},
    )


def _time_python(calls: int) -> dict[str, PythonTiming]:
    from sc_neurocore.transfer import (
        TransferConfig,
        apply_transfer_config,
        load_checkpoint,
        save_checkpoint,
    )

    checkpoint = _checkpoint_fixture()
    config = TransferConfig(freeze_until=0, lr_backbone=0.0, lr_head=0.01)

    with TemporaryDirectory(prefix="sc-transfer-bench-") as tmpdir:
        path = Path(tmpdir) / "model"
        start = perf_counter()
        for _ in range(calls):
            save_checkpoint(checkpoint, path)
            loaded = load_checkpoint(path)
            if loaded.total_params != checkpoint.total_params:
                raise RuntimeError("checkpoint roundtrip changed parameter count")
        roundtrip_seconds = perf_counter() - start

    start = perf_counter()
    for _ in range(calls):
        transfer_checkpoint = _checkpoint_fixture()
        _, rates = apply_transfer_config(transfer_checkpoint, config)
        if rates != [0.0, 0.01]:
            raise RuntimeError("transfer config produced unexpected learning rates")
    transfer_seconds = perf_counter() - start

    return {
        "checkpoint_roundtrip": {
            "calls": calls,
            "seconds": round(roundtrip_seconds, 6),
            "calls_per_second": round(calls / roundtrip_seconds, 3),
        },
        "apply_transfer_config": {
            "calls": calls,
            "seconds": round(transfer_seconds, 6),
            "calls_per_second": round(calls / transfer_seconds, 3),
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
    """Write transfer local evidence JSON and return a process status."""
    checkpoint_binary = Path("/tmp/sc_neurocore_transfer_checkpoint_tests")
    fine_tune_binary = Path("/tmp/sc_neurocore_transfer_fine_tune_tests")
    results: dict[str, object] = {
        "benchmark_id": "transfer-checkpoint-polyglot-local-validation",
        "evidence_class": "local_regression_non_isolated",
        "production_benchmark_claim": False,
        "python": _time_python(calls=100),
        "rust_checkpoint_compile": _run_command(
            [
                "rustc",
                "--edition=2021",
                "--test",
                "src/sc_neurocore/accel/rust/safety/checkpoint.rs",
                "-o",
                str(checkpoint_binary),
            ]
        ),
        "rust_checkpoint_tests": _run_command([str(checkpoint_binary)]),
        "rust_fine_tune_compile": _run_command(
            [
                "rustc",
                "--edition=2021",
                "--test",
                "src/sc_neurocore/accel/rust/safety/fine_tune.rs",
                "-o",
                str(fine_tune_binary),
            ]
        ),
        "rust_fine_tune_tests": _run_command([str(fine_tune_binary)]),
        "julia_validation": _run_command(
            [
                "julia",
                "--startup-file=no",
                "--history-file=no",
                "-e",
                (
                    'include("src/sc_neurocore/accel/julia/transfer/checkpoint.jl"); '
                    'include("src/sc_neurocore/accel/julia/transfer/fine_tune.jl"); '
                    "using .CheckpointAccel; using .FineTuneAccel; "
                    'state = SNNCheckpointState([[0.1 0.2; 0.3 0.4], [0.5 0.6]], ["hidden", "output"], '
                    '[(2, 2), (2, 1)], ["LIF", "LIF"], ["hidden"]); '
                    "@assert validate_checkpoint(state); @assert total_params(state) == 6; "
                    'ckpt = TransferCheckpointState(["hidden", "output"]); '
                    "rates = apply_transfer_config!(ckpt, TransferConfigState(0, 0.0, 0.01)); "
                    "@assert rates == [0.0, 0.01]"
                ),
            ]
        ),
        "mojo_checkpoint_validation": _run_command(
            ["mojo", "src/sc_neurocore/accel/mojo/kernels/checkpoint.mojo"]
        ),
        "mojo_fine_tune_validation": _run_command(
            ["mojo", "src/sc_neurocore/accel/mojo/kernels/fine_tune.mojo"]
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
