#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — controlled Wu et al. IQIF five-backend benchmark

"""Measure the pinned IQIF tutorial through every public batch dispatcher.

The report binds real call timings to bit-exact signed-integer trajectories,
event counts, final state, source and loaded-binary hashes, CPU affinity,
runtime versions, and an executable standalone Rust-safety gate. Missing
backends, partial parity, or unacknowledged unpinned execution fail closed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import iqif as backends
from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_STEPS = 200_000
N_REPEATS = 7
WARMUP_STEPS = 1_000
CURRENT = 10
KERNEL = "iqif_integer_q03_batch"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
SOURCE_PATHS = (
    "benchmarks/bench_model_iqif.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/lib.rs",
    "engine/src/network_runner.rs",
    "engine/src/neurons/trivial/integer_qif.rs",
    "engine/src/pyo3_neurons.rs",
    "src/sc_neurocore/accel/iqif.py",
    "src/sc_neurocore/accel/go/neurons/iqif/iqif.go",
    "src/sc_neurocore/accel/go/neurons/iqif/libiqif.h",
    "src/sc_neurocore/accel/go/services/iqif.go",
    "src/sc_neurocore/accel/go/services/iqif_test.go",
    "src/sc_neurocore/accel/julia/neurons/iqif.jl",
    "src/sc_neurocore/accel/mojo/kernels/iqif.mojo",
    "src/sc_neurocore/accel/rust/safety/iqif.rs",
    "src/sc_neurocore/neurons/model_schemas/iqif.json",
    "src/sc_neurocore/neurons/model_schemas/iqif.toml",
    "src/sc_neurocore/neurons/models/iqif.py",
)
GO_LIBRARY = REPOSITORY / "src/sc_neurocore/accel/go/neurons/iqif/libiqif.so"
MOJO_LIBRARY = REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/libiqif.so"


def _configured() -> IntegerQIFNeuron:
    """Return the exact pinned-source tutorial state and parameters."""
    return IntegerQIFNeuron()


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


def _source_hashes() -> dict[str, object]:
    """Hash every implementation and ABI surface relevant to this closure."""
    flat = {
        relative: hashlib.sha256((REPOSITORY / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }
    nested: dict[str, object] = {}
    for relative, digest in flat.items():
        stem, suffix = relative.rsplit(".", 1)
        grouped = nested.setdefault(stem, {})
        if not isinstance(grouped, dict):
            raise RuntimeError(f"source hash namespace collision at {stem}")
        grouped[suffix] = digest
    return {**flat, **nested}


def _binary_hashes() -> dict[str, dict[str, str | int]]:
    """Bind the exact native artifacts loaded by the measured dispatchers."""
    native = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    native_file = getattr(native, "__file__", None)
    if not isinstance(native_file, str):
        raise RuntimeError("loaded Rust extension does not expose a filesystem path")
    binaries = {
        "rust_extension": Path(native_file),
        "go_shared_library": GO_LIBRARY,
        "mojo_shared_library": MOJO_LIBRARY,
    }
    records: dict[str, dict[str, str | int]] = {}
    repository = REPOSITORY.resolve()
    for name, path in binaries.items():
        resolved = path.resolve(strict=True)
        try:
            display_path = resolved.relative_to(repository).as_posix()
        except ValueError:
            display_path = str(resolved)
        records[name] = {
            "path": display_path,
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "size_bytes": resolved.stat().st_size,
        }
    return records


def _trace_digest(trace: npt.NDArray[np.int64]) -> str:
    """Return a platform-stable SHA-256 over signed little-endian states."""
    return hashlib.sha256(np.asarray(trace, dtype="<i8").tobytes()).hexdigest()


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Return backend availability plus a deterministic diagnostic."""
    if backend == "python":
        return True, ""
    if backend == "rust":
        available = backends._HAS_RUST
        return available, "" if available else "Rust engine IQIF batch unavailable"
    if backend == "julia":
        available = backends.ensure_julia_loaded()
        return available, "" if available else "juliacall or IQIF module unavailable"
    if backend == "go":
        available = backends.ensure_go_loaded()
        return available, "" if available else "compiled Go libiqif.so unavailable"
    available = backends.ensure_mojo_loaded()
    return available, "" if available else "compiled Mojo libiqif.so unavailable"


def _measure_backend(
    backend: str,
) -> tuple[
    float,
    float,
    float,
    list[float],
    npt.NDArray[np.int64],
    int,
    int,
]:
    """Warm one public lane, then return timing and exact final state."""
    _configured().simulate(min(WARMUP_STEPS, N_STEPS), CURRENT, backend=backend)
    elapsed_ms: list[float] = []
    trace: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
    spikes = 0
    final_v = 128
    for _repeat in range(N_REPEATS):
        gc.collect()
        neuron = _configured()
        started = time.perf_counter_ns()
        trace, spikes = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        final_v = neuron.v
    return (
        statistics.median(elapsed_ms),
        min(elapsed_ms),
        max(elapsed_ms),
        elapsed_ms,
        trace,
        spikes,
        final_v,
    )


def _verify_rust_safety() -> dict[str, Any]:
    """Compile and execute the actual standalone IQIF Rust-safety tests."""
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/iqif.rs"
    with tempfile.TemporaryDirectory(prefix="sc_neurocore_iqif_safety_") as temp_dir:
        binary = Path(temp_dir) / "iqif_safety_tests"
        compile_command = [
            "rustc",
            "--edition=2021",
            "--test",
            str(source),
            "-O",
            "-o",
            str(binary),
        ]
        try:
            compiled = subprocess.run(
                compile_command,
                cwd=REPOSITORY,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if compiled.returncode == 0:
                executed = subprocess.run(
                    [str(binary)],
                    cwd=REPOSITORY,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            else:
                executed = compiled
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "command": " ".join(compile_command) + " && <compiled-test-binary>",
                "passed": False,
                "returncode": -1,
                "output_tail": [str(exc)],
            }
    output = (executed.stdout + "\n" + executed.stderr).strip().splitlines()
    return {
        "command": " ".join(compile_command[:-2]) + " -o <temp-binary> && <temp-binary>",
        "passed": compiled.returncode == 0 and executed.returncode == 0,
        "returncode": executed.returncode if compiled.returncode == 0 else compiled.returncode,
        "output_tail": output[-12:],
    }


def _runtime_versions() -> dict[str, str]:
    """Record every runtime involved in the measured closure."""
    home = Path.home()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rust": _tool_version([_tool_path("rustc") or "", "--version"]),
        "julia": _tool_version(
            [_tool_path("julia", home / ".juliaup/bin/julia") or "", "--version"]
        ),
        "go": _tool_version([_tool_path("go") or "", "version"]),
        "mojo": _tool_version([_tool_path("mojo") or "", "--version"]),
    }


def _environment(load_start: tuple[float, float, float]) -> dict[str, Any]:
    """Capture affinity and load without overstating exclusive isolation."""
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
        "exclusive_cpu_isolation_claimed": False,
        "kernel_isolated_cpus": _read_optional(Path("/sys/devices/system/cpu/isolated")),
        "kernel_nohz_full_cpus": _read_optional(Path("/sys/devices/system/cpu/nohz_full")),
        "governor": governor,
        "load_average_start": list(load_start),
        "load_average_end": list(os.getloadavg()),
        "measurement_scope": (
            "single-logical-CPU taskset affinity; CPU is not claimed exclusively isolated"
        ),
        "runtime_versions": _runtime_versions(),
    }


def main(argv: list[str]) -> int:
    """Run the controlled benchmark and write its evidence artefact."""
    parser = argparse.ArgumentParser(description="Controlled IQIF five-backend benchmark")
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
    reference: npt.NDArray[np.int64] | None = None
    reference_ms: float | None = None
    reference_spikes: int | None = None
    reference_final_v: int | None = None
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            rows[backend] = {"available": False, "used": False, "unavailable_reason": reason}
            continue
        (
            median_ms,
            minimum_ms,
            maximum_ms,
            samples_ms,
            trace,
            spikes,
            final_v,
        ) = _measure_backend(backend)
        if backend == "python":
            reference = trace
            reference_ms = median_ms
            reference_spikes = spikes
            reference_final_v = final_v
            mismatch_count = 0
        else:
            if (
                reference is None
                or reference_ms is None
                or reference_spikes is None
                or reference_final_v is None
            ):
                raise RuntimeError("Python reference must be measured first")
            mismatch_count = int(np.count_nonzero(trace != reference))
        rows[backend] = {
            "available": True,
            "used": True,
            "median_call_ms": median_ms,
            "minimum_call_ms": minimum_ms,
            "maximum_call_ms": maximum_ms,
            "samples_call_ms": samples_ms,
            "median_ns_per_step": median_ms * 1_000_000.0 / N_STEPS,
            "speedup_vs_python": reference_ms / median_ms if reference_ms is not None else 1.0,
            "trace_mismatch_count": mismatch_count,
            "parity_max_abs_diff": (
                0 if mismatch_count == 0 else int(np.max(np.abs(trace - reference)))
            ),
            "trace_matches_python": mismatch_count == 0,
            "event_count": spikes,
            "event_count_matches_python": (
                True if reference_spikes is None else spikes == reference_spikes
            ),
            "final_state_matches_python": (
                True if reference_final_v is None else final_v == reference_final_v
            ),
            "trace_sha256": _trace_digest(trace),
            "final_state": {"v": final_v},
        }

    measured_order = sorted(
        (backend for backend in BACKENDS if rows[backend].get("used") is True),
        key=lambda backend: float(rows[backend]["median_call_ms"]),
    )
    native_order = [backend for backend in measured_order if backend != "python"]
    auto_order = [*native_order, "python"]
    rust_safety = _verify_rust_safety()
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "kernel": KERNEL,
        "evidence_class": "local_regression_single_cpu_affinity_non_exclusive",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": (
            "taskset --cpu-list <cpu> env PYTHONPATH=bridge:src:. .venv/bin/python "
            "benchmarks/bench_model_iqif.py --json <artifact>"
        ),
        "workload": {
            "n_steps": N_STEPS,
            "repeats": N_REPEATS,
            "warmup_steps": WARMUP_STEPS,
            "current": CURRENT,
            "initial_state": {
                "v": 128,
                "v_rest": 128,
                "v_threshold": 200,
                "v_reset": 128,
                "a": 1,
                "b": 1,
                "v_max": 255,
                "v_min": 0,
            },
            "parameters": "pinned Wu et al. tutorial defaults; complete signed-int32 ABI",
        },
        "meta": _environment(load_start),
        "backends": rows,
        "measured_order": measured_order,
        "fastest_measured_native_backend": native_order[0] if native_order else "python",
        "auto_backend_order": auto_order,
        "recommended_auto_backend": auto_order[0],
        "auto_backend_selection_basis": (
            "same-host measured warm batch order; non-exclusive timings remain diagnostic"
        ),
        "verification": {"rust_safety": rust_safety},
        "source_hashes": _source_hashes(),
        "binary_hashes": _binary_hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"IQIF benchmark: {N_STEPS} steps x {N_REPEATS} repeats")
    for backend in measured_order:
        row = rows[backend]
        print(
            f"{backend:>7}: {float(row['median_call_ms']):10.3f} ms  "
            f"{float(row['speedup_vs_python']):8.2f}x  "
            f"mismatches={int(row['trace_mismatch_count'])}  "
            f"events={int(row['event_count'])}  v={int(row['final_state']['v'])}"
        )
    print(f"Measured order: {', '.join(measured_order)}")
    print(f"Auto backend order: {', '.join(auto_order)}")
    print(f"Rust safety tests: {'PASS' if rust_safety['passed'] else 'FAIL'}")
    print(f"Wrote {args.json}")

    if not rust_safety["passed"]:
        return 5
    if any(
        not bool(row.get("trace_matches_python", True))
        or not bool(row.get("event_count_matches_python", True))
        or not bool(row.get("final_state_matches_python", True))
        for row in rows.values()
    ):
        return 3
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
