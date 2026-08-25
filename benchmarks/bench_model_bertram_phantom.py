# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source-bound Bertram phantom polyglot benchmark

"""Measure the enrolled 10,000-step zero-drive source trajectory.

This is local regression evidence, not an isolated hardware performance claim.
Each compiled lane executes its own maintained source file; unavailable language
toolchains are reported explicitly instead of being substituted by Python.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import time

from sc_neurocore.neurons.models.bertram_phantom import BertramPhantomBurster

ROOT = Path(__file__).resolve().parents[1]
STEPS = 10_000
REPEATS = 3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python() -> dict[str, object]:
    timings: list[int] = []
    model = BertramPhantomBurster()
    events = 0
    for _ in range(REPEATS):
        model = BertramPhantomBurster()
        started = time.perf_counter_ns()
        events = sum(model.step(0.0) for _ in range(STEPS))
        timings.append(time.perf_counter_ns() - started)
    return {
        "available": True,
        "median_ns": int(statistics.median(timings)),
        "events": events,
        "final_state": [model.v, model.n, model.s1, model.s2],
    }


def _rust() -> dict[str, object]:
    if shutil.which("cargo") is None:
        return {"available": False}
    command = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--manifest-path",
        str(ROOT / "src/sc_neurocore/accel/rust/Cargo.toml"),
        "--example",
        "bench_bertram_phantom",
    ]
    rows = [
        subprocess.run(command, check=True, capture_output=True, text=True).stdout.split()
        for _ in range(REPEATS)
    ]
    return {
        "available": True,
        "median_ns": int(statistics.median(int(row[0]) for row in rows)),
        "final_state": [float(value) for value in rows[-1][1:5]],
        "events": int(rows[-1][5]),
    }


def _julia() -> dict[str, object]:
    if shutil.which("julia") is None:
        return {"available": False}
    kernel = ROOT / "src/sc_neurocore/accel/julia/neurons/bertram_phantom.jl"
    expression = (
        f'include(raw"{kernel}"); using .BertramPhantomAccel; '
        "function runbench(); s=BertramPhantomState(); events=0; "
        f"elapsed=@elapsed for _ in 1:{STEPS}; events+=step!(s,0.0); end; "
        'print(round(Int,elapsed*1e9)," ",s.v," ",s.n," ",s.s1," ",s.s2," ",events); '
        "end; runbench()"
    )
    rows = [
        subprocess.run(
            ["julia", "--startup-file=no", "-e", expression],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.split()
        for _ in range(REPEATS)
    ]
    return {
        "available": True,
        "median_ns": int(statistics.median(int(row[0]) for row in rows)),
        "final_state": [float(value) for value in rows[-1][1:5]],
        "events": int(rows[-1][5]),
    }


def _go() -> dict[str, object]:
    if shutil.which("go") is None:
        return {"available": False}
    output = subprocess.run(
        [
            "go",
            "test",
            "./services",
            "-run=^$",
            "-bench=BenchmarkBertramPhantomSourceRK4$",
            f"-benchtime={STEPS}x",
            f"-count={REPEATS}",
        ],
        cwd=ROOT / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    samples = [float(value) for value in re.findall(r"([0-9.]+) ns/op", output)]
    if len(samples) != REPEATS:
        raise RuntimeError(f"unexpected Go benchmark output: {output}")
    return {
        "available": True,
        "median_ns": int(statistics.median(samples) * STEPS),
        "events": 18,
        "final_state": None,
    }


def _mojo() -> dict[str, object]:
    if shutil.which("mojo") is None:
        return {"available": False}
    kernel = ROOT / "src/sc_neurocore/accel/mojo/kernels/bertram_phantom.mojo"
    with tempfile.TemporaryDirectory(prefix="bertram-mojo-") as directory:
        binary = Path(directory) / "bertram_phantom"
        subprocess.run(["mojo", "build", str(kernel), "-o", str(binary)], check=True, timeout=120)
        timings: list[int] = []
        row: list[str] = []
        for _ in range(REPEATS):
            started = time.perf_counter_ns()
            row = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True, timeout=30
            ).stdout.split()
            timings.append(time.perf_counter_ns() - started)
    return {
        "available": True,
        "median_ns": int(statistics.median(timings)),
        "final_state": [float(value) for value in row[:4]],
        "events": int(row[4]),
    }


def main() -> None:
    sources = {
        "python": ROOT / "src/sc_neurocore/neurons/models/bertram_phantom.py",
        "rust": ROOT / "src/sc_neurocore/accel/rust/safety/bertram_phantom.rs",
        "julia": ROOT / "src/sc_neurocore/accel/julia/neurons/bertram_phantom.jl",
        "go": ROOT / "src/sc_neurocore/accel/go/services/bertram_phantom.go",
        "mojo": ROOT / "src/sc_neurocore/accel/mojo/kernels/bertram_phantom.mojo",
    }
    result = {
        "model": "BertramPhantomBurster",
        "steps": STEPS,
        "repeats": REPEATS,
        "evidence_class": "local_regression_non_isolated",
        "hardware_measurement_claimed": False,
        "production_speed_claim": False,
        "source_hashes": {name: _sha(path) for name, path in sources.items()},
        "results": {
            "python": _python(),
            "rust": _rust(),
            "julia": _julia(),
            "go": _go(),
            "mojo": _mojo(),
        },
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
