# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled AdEx five-backend benchmark

"""Measure the maintained AdEx baseline-Euler recurrence in every accel lane.

The evidence record binds timings to source hashes, CPU affinity, host load,
runtime versions, event parity and the measured voltage-trace error. Missing
backends or unpinned execution fail unless the operator explicitly permits the
diagnostic condition.
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
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import sc_neurocore.neurons.models.adex as adex
from sc_neurocore.neurons.models.adex import AdExNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_STEPS = 100_000
N_REPEATS = 7
CURRENT = 500.0
TRACE_ATOL = 5.0e-12
KERNEL = "adex_baseline_euler_simulate"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
SOURCE_PATHS = (
    "benchmarks/bench_adex.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/adex_neuron.rs",
    "engine/src/neuron/adex.rs",
    "engine/src/network_runner/model_catalogue.rs",
    "engine/src/network_runner/model_factory.rs",
    "engine/src/network_runner/neuron_variant.rs",
    "hdl/formal/catalogue/sc_adex.sby",
    "hdl/formal/catalogue/sc_adex.v",
    "hdl/formal/catalogue/sc_adex_formal.v",
    "hdl/reports/yosys_adex_q1616_2026-08-30.json",
    "src/sc_neurocore/accel/go/neurons/adex/adex.go",
    "src/sc_neurocore/accel/go/services/adex.go",
    "src/sc_neurocore/accel/julia/neurons/adex.jl",
    "src/sc_neurocore/accel/mojo/kernels/adex.mojo",
    "src/sc_neurocore/accel/rust/safety/adex.rs",
    "src/sc_neurocore/neurons/model_descriptors/AdExNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/adex.json",
    "src/sc_neurocore/neurons/model_schemas/adex.toml",
    "src/sc_neurocore/neurons/models/adex.py",
    "src/sc_neurocore/neurons/reference_receipts/adex_brette_gerstner_2005.json",
    "src/sc_neurocore/neurons/reference_trace_data/adex_resting_adaptation_doi.json",
)


def _cpu_model() -> str:
    """Return the first Linux CPU model string, or a portable fallback."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _read_optional(path: Path) -> str:
    """Read one host metadata file without making it a dependency."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _tool_path(name: str, fallback: Path | None = None) -> str | None:
    """Resolve a runtime executable with one explicit fallback."""
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    if fallback is not None and fallback.is_file():
        return str(fallback)
    return None


def _tool_version(command: list[str]) -> str:
    """Return the first version line for a runtime executable."""
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
    """Hash every implementation and ABI surface relevant to this closure."""
    return {
        relative: hashlib.sha256((REPOSITORY / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }


def _trace_digest(
    v_trace: npt.NDArray[np.float64],
    w_trace: npt.NDArray[np.float64],
    event_trace: npt.NDArray[np.uint8],
) -> str:
    """Digest the complete aligned state/event packet."""
    payload = b"".join(
        struct.pack("<ddB", float(v), float(w), int(event))
        for v, w, event in zip(v_trace, w_trace, event_trace, strict=True)
    )
    return hashlib.sha256(payload).hexdigest()


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Return backend availability plus a deterministic diagnostic."""
    if backend == "python":
        return True, ""
    if backend == "rust":
        return adex._HAS_RUST, "" if adex._HAS_RUST else "Rust engine AdEx symbol unavailable"
    if backend == "julia":
        available = adex._ensure_julia_loaded()
        return available, "" if available else "juliacall or AdEx Julia module unavailable"
    if backend == "go":
        available = adex._ensure_go_loaded()
        return available, "" if available else "compiled Go libadex.so unavailable"
    available = adex._ensure_mojo_loaded()
    return available, "" if available else "compiled Mojo libadex.so unavailable"


def _measure_backend(
    backend: str,
) -> tuple[
    float,
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
    int,
    tuple[float, float],
]:
    """Warm one backend, then return timings and final numerical state."""
    AdExNeuron().simulate_complete(20, CURRENT, backend=backend)
    elapsed_ms: list[float] = []
    v_trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    w_trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    event_trace: npt.NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
    spikes = 0
    final_state = (0.0, 0.0)
    for _repeat in range(N_REPEATS):
        gc.collect()
        neuron = AdExNeuron()
        started = time.perf_counter_ns()
        v_trace, w_trace, event_trace = neuron.simulate_complete(N_STEPS, CURRENT, backend=backend)
        spikes = int(np.sum(event_trace, dtype=np.int64))
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        final_state = (neuron.v, neuron.w)
    return (
        statistics.median(elapsed_ms),
        min(elapsed_ms),
        v_trace,
        w_trace,
        event_trace,
        spikes,
        final_state,
    )


def _runtime_versions() -> dict[str, str]:
    """Record all runtimes involved in the measured closure."""
    home = Path.home()
    mojo = _tool_path("mojo", home / ".pixi/bin/mojo") or ""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rust": _tool_version([_tool_path("rustc") or "", "--version"]),
        "julia": _tool_version([_tool_path("julia") or "", "--version"]),
        "go": _tool_version([_tool_path("go") or "", "version"]),
        "mojo": _tool_version([mojo, "--version"]),
    }


def _environment(load_start: tuple[float, float, float]) -> dict[str, Any]:
    """Capture affinity and load without claiming kernel isolation."""
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
    """Run the controlled benchmark and write its evidence artefact."""
    parser = argparse.ArgumentParser(description="Controlled AdEx five-backend benchmark")
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
    reference: (
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.uint8],
        ]
        | None
    ) = None
    reference_ms: float | None = None
    reference_spikes: int | None = None
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            rows[backend] = {
                "available": False,
                "used": False,
                "unavailable_reason": reason,
            }
            continue
        (
            median_ms,
            minimum_ms,
            v_trace,
            w_trace,
            event_trace,
            spikes,
            final_state,
        ) = _measure_backend(backend)
        if backend == "python":
            reference = (v_trace, w_trace, event_trace)
            reference_ms = median_ms
            reference_spikes = spikes
            v_parity = 0.0
            w_parity = 0.0
        else:
            if reference is None or reference_ms is None or reference_spikes is None:
                raise RuntimeError("Python reference must be measured first")
            reference_v, reference_w, _reference_events = reference
            v_parity = float(np.max(np.abs(v_trace - reference_v))) if v_trace.size else 0.0
            w_parity = float(np.max(np.abs(w_trace - reference_w))) if w_trace.size else 0.0
        rows[backend] = {
            "available": True,
            "used": True,
            "median_call_ms": median_ms,
            "minimum_call_ms": minimum_ms,
            "speedup_vs_python": (reference_ms / median_ms) if reference_ms is not None else 1.0,
            "v_parity_max_abs_diff": v_parity,
            "w_parity_max_abs_diff": w_parity,
            "event_count": spikes,
            "event_count_matches_python": (
                True if reference_spikes is None else spikes == reference_spikes
            ),
            "event_trace_matches_python": (
                True if reference is None else bool(np.array_equal(event_trace, reference[2]))
            ),
            "complete_packet_sha256": _trace_digest(v_trace, w_trace, event_trace),
            "final_state": dict(zip(("v", "w"), final_state, strict=True)),
        }

    measured_order = sorted(
        (backend for backend in BACKENDS if rows[backend].get("used") is True),
        key=lambda backend: float(rows[backend]["median_call_ms"]),
    )
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v2",
        "kernel": KERNEL,
        "workload": {
            "n_steps": N_STEPS,
            "repeats": N_REPEATS,
            "current": CURRENT,
            "parameters": "AdEx maintained defaults; candidate-first baseline Euler",
            "state_trace_atol": TRACE_ATOL,
        },
        "meta": _environment(load_start),
        "backends": rows,
        "measured_order": measured_order,
        "source_hashes": _source_hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"AdEx benchmark: {N_STEPS} steps x {N_REPEATS} repeats")
    for backend in measured_order:
        row = rows[backend]
        print(
            f"{backend:>7}: {float(row['median_call_ms']):10.3f} ms  "
            f"{float(row['speedup_vs_python']):8.2f}x  "
            f"max|dv|={float(row['v_parity_max_abs_diff']):.3e}  "
            f"max|dw|={float(row['w_parity_max_abs_diff']):.3e}  "
            f"events={int(row['event_count'])}"
        )
    print(f"Measured order: {', '.join(measured_order)}")
    print(f"Wrote {args.json}")

    if any(not bool(row.get("event_count_matches_python", True)) for row in rows.values()):
        return 3
    if any(not bool(row.get("event_trace_matches_python", True)) for row in rows.values()):
        return 3
    if any(
        max(
            float(row.get("v_parity_max_abs_diff", 0.0)),
            float(row.get("w_parity_max_abs_diff", 0.0)),
        )
        > TRACE_ATOL
        for row in rows.values()
        if row.get("used") is True
    ):
        return 4
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
