# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled cross-language Chialvo map benchmark

"""Benchmark every production Chialvo simulation lane on one logical CPU.

The JSON output is both a measurement record and the data source for the
host-matched fastest-first dispatcher. Availability, affinity, kernel CPU
isolation, governor, load, runtime versions, parity, and source hashes are
recorded explicitly. Missing backends or unpinned execution fail by default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models import chialvo_map
from sc_neurocore.neurons.models.chialvo_map import ChialvoMapNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_STEPS = 500_000
N_REPEATS = 5
CURRENT = 0.0
KERNEL = "chialvo_map_simulate"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
PARITY_ATOL = {
    "python": 0.0,
    "rust": 1.0e-10,
    "julia": 1.0e-10,
    "go": 1.0e-10,
    "mojo": 1.0e-6,
}
SOURCE_PATHS = (
    "benchmarks/bench_chialvo_map.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/neurons/chialvo_map.rs",
    "src/sc_neurocore/accel/go/neurons/chialvo_map/chialvo_map.go",
    "src/sc_neurocore/accel/go/services/chialvo_map.go",
    "src/sc_neurocore/accel/julia/neurons/chialvo_map.jl",
    "src/sc_neurocore/accel/mojo/neurons/chialvo_map.mojo",
    "src/sc_neurocore/accel/rust/safety/chialvo_map.rs",
    "src/sc_neurocore/neurons/models/chialvo_map.py",
)


def _cpu_model() -> str:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _tool_path(name: str, fallback: Path | None = None) -> str | None:
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    if fallback is not None and fallback.is_file():
        return str(fallback)
    return None


def _tool_version(command: list[str]) -> str:
    if not command or command[0] == "":
        return "unavailable"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else f"exit {result.returncode}"


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        payload = (REPOSITORY / relative).read_bytes()
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    return hashes


def _probe_backend(backend: str) -> tuple[bool, str]:
    if backend == "python":
        return True, ""
    if backend == "rust":
        available = chialvo_map._HAS_RUST
        return available, "" if available else "engine batch function unavailable"
    if backend == "julia":
        available = chialvo_map._ensure_julia_loaded()
        return available, "" if available else "juliacall or Julia module unavailable"
    if backend == "go":
        available = chialvo_map._ensure_go_loaded()
        return available, "" if available else "libchialvo.so Go build unavailable"
    available = chialvo_map._ensure_mojo_loaded()
    return available, "" if available else "libchialvo.so Mojo build unavailable"


def _measure_backend(
    backend: str,
) -> tuple[float, float, npt.NDArray[np.float64], int, float, float]:
    ChialvoMapNeuron().simulate(2_000, CURRENT, backend=backend)
    elapsed_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    spikes = 0
    x_final = 0.0
    y_final = 0.0
    for _repeat in range(N_REPEATS):
        gc.collect()
        neuron = ChialvoMapNeuron()
        started = time.perf_counter_ns()
        trace, spikes = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        x_final, y_final = neuron.x, neuron.y
    return (
        statistics.median(elapsed_ms),
        min(elapsed_ms),
        trace,
        spikes,
        x_final,
        y_final,
    )


def _runtime_versions() -> dict[str, str]:
    home = Path.home()
    julia = _tool_path("julia", home / ".juliaup/bin/julia") or ""
    mojo = _tool_path("mojo", home / ".pixi/bin/mojo") or ""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rust": _tool_version([_tool_path("rustc") or "", "--version"]),
        "go": _tool_version([_tool_path("go") or "", "version"]),
        "julia": _tool_version([julia, "--version"]),
        "mojo": _tool_version([mojo, "--version"]),
    }


def _environment(load_start: tuple[float, float, float]) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    cpu = affinity[0] if len(affinity) == 1 else None
    governor = (
        _read_optional(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"))
        if cpu is not None
        else "mixed-or-unpinned"
    )
    return {
        "cpu": _cpu_model(),
        "platform": platform.platform(),
        "affinity": affinity,
        "single_cpu_pinned": len(affinity) == 1,
        "kernel_isolated_cpus": _read_optional(Path("/sys/devices/system/cpu/isolated")),
        "governor": governor,
        "load_average_start": list(load_start),
        "load_average_end": list(os.getloadavg()),
        "measurement_scope": (
            "single-logical-CPU affinity; kernel isolation and workstation load reported separately"
        ),
        "runtime_versions": _runtime_versions(),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Controlled all-language Chialvo map benchmark")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument("--allow-unavailable-backends", action="store_true")
    args = parser.parse_args(argv)

    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1 and not args.allow_unpinned:
        print(f"Refusing unpinned benchmark; affinity is {affinity}")
        return 2

    load_start = os.getloadavg()
    probes = {backend: _probe_backend(backend) for backend in BACKENDS}
    missing = [backend for backend, (available, _reason) in probes.items() if not available]
    if missing and not args.allow_unavailable_backends:
        print("Missing required backend(s): " + ", ".join(missing))
        return 2

    rows: dict[str, dict[str, Any]] = {}
    reference: npt.NDArray[np.float64] | None = None
    reference_ms: float | None = None
    reference_spikes: int | None = None
    reference_final: tuple[float, float] | None = None
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            rows[backend] = {
                "available": False,
                "used": False,
                "unavailable_reason": reason,
            }
            continue
        median_ms, minimum_ms, trace, spikes, x_final, y_final = _measure_backend(backend)
        if backend == "python":
            reference = trace
            reference_ms = median_ms
            reference_spikes = spikes
            reference_final = (x_final, y_final)
            parity = 0.0
            final_state_delta = 0.0
        else:
            if (
                reference is None
                or reference_ms is None
                or reference_spikes is None
                or reference_final is None
            ):
                raise RuntimeError("Python reference must be measured first")
            parity = float(np.max(np.abs(trace - reference)))
            final_state_delta = max(
                abs(x_final - reference_final[0]),
                abs(y_final - reference_final[1]),
            )
        rows[backend] = {
            "available": True,
            "used": True,
            "median_call_ms": median_ms,
            "minimum_call_ms": minimum_ms,
            "speedup_vs_python": (reference_ms / median_ms) if reference_ms is not None else 1.0,
            "parity_max_abs_diff": parity,
            "parity_atol": PARITY_ATOL[backend],
            "event_count": spikes,
            "event_count_matches_python": (
                True if reference_spikes is None else spikes == reference_spikes
            ),
            "final_state": {"x": x_final, "y": y_final},
            "final_state_max_abs_diff": final_state_delta,
            "final_state_matches_python": final_state_delta <= PARITY_ATOL[backend],
        }

    measured_order = sorted(
        (backend for backend in BACKENDS if rows[backend].get("used") is True),
        key=lambda backend: float(rows[backend]["median_call_ms"]),
    )
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "kernel": KERNEL,
        "workload": {
            "n_steps": N_STEPS,
            "repeats": N_REPEATS,
            "current": CURRENT,
            "parameters": "Chialvo source-paper defaults",
        },
        "meta": _environment(load_start),
        "backends": rows,
        "measured_order": measured_order,
        "source_hashes": _source_hashes(),
    }
    print(f"Chialvo benchmark: {N_STEPS:,} iterations x {N_REPEATS} repeats")
    for backend in measured_order:
        row = rows[backend]
        print(
            f"{backend:>7}: {float(row['median_call_ms']):10.3f} ms  "
            f"{float(row['speedup_vs_python']):8.2f}x  "
            f"max|delta|={float(row['parity_max_abs_diff']):.3e}  "
            f"events={int(row['event_count'])}"
        )
    print(f"Measured order: {', '.join(measured_order)}")
    fidelity_failed = any(
        row.get("used") is True
        and (
            not bool(row.get("event_count_matches_python"))
            or not bool(row.get("final_state_matches_python"))
            or float(row["parity_max_abs_diff"]) > PARITY_ATOL[backend]
        )
        for backend, row in rows.items()
    )
    if fidelity_failed:
        print("Chialvo fidelity failed; benchmark evidence was not written")
        return 3
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
