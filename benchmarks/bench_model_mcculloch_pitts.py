#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — controlled McCulloch-Pitts five-backend benchmark

"""Measure the source-faithful varying-input rule through every dispatcher.

The report binds real public-call timings to exact binary traces, counts,
input/source/binary hashes, CPU affinity, runtime versions and an executable
standalone Rust-safety gate. Timings are local non-exclusive regression
evidence, not production throughput or hardware measurements.
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
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import mcculloch_pitts as backends
from sc_neurocore.neurons.models.mcculloch_pitts import McCullochPittsNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_ROWS = 200_000
N_REPEATS = 7
WARMUP_ROWS = 1_000
THETA = 7
KERNEL = "mcculloch_pitts_absolute_inhibition_batch"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
SOURCE_PATHS = (
    "benchmarks/bench_model_mcculloch_pitts.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/mcculloch_pitts.rs",
    "engine/src/network_runner.rs",
    "engine/src/neurons/rate/mcculloch_pitts.rs",
    "src/sc_neurocore/accel/mcculloch_pitts.py",
    "src/sc_neurocore/accel/go/neurons/mcculloch_pitts/mcculloch_pitts.go",
    "src/sc_neurocore/accel/go/neurons/mcculloch_pitts/libmcculloch_pitts.h",
    "src/sc_neurocore/accel/go/services/mcculloch_pitts.go",
    "src/sc_neurocore/accel/go/services/mcculloch_pitts_test.go",
    "src/sc_neurocore/accel/julia/neurons/mcculloch_pitts.jl",
    "src/sc_neurocore/accel/mojo/kernels/mcculloch_pitts.mojo",
    "src/sc_neurocore/accel/rust/safety/mcculloch_pitts.rs",
    "src/sc_neurocore/neurons/model_descriptors/McCullochPittsNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/mcculloch_pitts.json",
    "src/sc_neurocore/neurons/model_schemas/mcculloch_pitts.toml",
    "src/sc_neurocore/neurons/models/mcculloch_pitts.py",
    "src/sc_neurocore/neurons/reference_trace_data/mcculloch_pitts_1943_truth_table.json",
)
GO_LIBRARY = REPOSITORY / "src/sc_neurocore/accel/go/neurons/mcculloch_pitts/libmcculloch_pitts.so"
MOJO_LIBRARY = REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/libmcculloch_pitts.so"


def _inputs(n_rows: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Return deterministic threshold, equality, maximum and veto coverage."""
    indices = np.arange(n_rows, dtype=np.int64)
    counts = np.ascontiguousarray(indices % 16, dtype=np.int64)
    if n_rows > 0:
        counts[-1] = (1 << 31) - 1
    flags = np.ascontiguousarray(indices % 11 == 0, dtype=np.bool_)
    return counts, flags


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
    """Bind the exact native artifacts loaded by measured dispatchers."""
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


def _trace_digest(trace: npt.NDArray[np.uint8]) -> str:
    """Return a platform-stable SHA-256 over exact binary events."""
    return hashlib.sha256(np.asarray(trace, dtype=np.uint8).tobytes()).hexdigest()


def _input_digests(
    counts: npt.NDArray[np.int64],
    flags: npt.NDArray[np.bool_],
) -> dict[str, str]:
    """Bind the deterministic varying-input workload byte-for-byte."""
    return {
        "excitatory_counts_sha256": hashlib.sha256(
            np.asarray(counts, dtype="<i8").tobytes()
        ).hexdigest(),
        "inhibitory_flags_sha256": hashlib.sha256(
            np.asarray(flags, dtype=np.uint8).tobytes()
        ).hexdigest(),
    }


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Return backend availability plus a deterministic diagnostic."""
    if backend == "python":
        return True, ""
    if backend == "rust":
        available = backends._HAS_RUST
        return available, "" if available else "Rust McCulloch-Pitts batch unavailable"
    if backend == "julia":
        available = backends.ensure_julia_loaded()
        return available, "" if available else "juliacall or Julia module unavailable"
    if backend == "go":
        available = backends.ensure_go_loaded()
        return available, "" if available else "compiled Go library unavailable"
    available = backends.ensure_mojo_loaded()
    return available, "" if available else "compiled Mojo library unavailable"


def _measure_backend(
    backend: str,
) -> tuple[float, float, float, list[float], npt.NDArray[np.uint8], int]:
    """Warm one public lane, then return timing and exact binary output."""
    warm_counts, warm_flags = _inputs(min(WARMUP_ROWS, N_ROWS))
    McCullochPittsNeuron(theta=THETA).simulate(
        warm_counts,
        warm_flags,
        backend=backend,
    )
    counts, flags = _inputs(N_ROWS)
    elapsed_ms: list[float] = []
    trace: npt.NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
    event_count = 0
    for _repeat in range(N_REPEATS):
        gc.collect()
        started = time.perf_counter_ns()
        trace, event_count = McCullochPittsNeuron(theta=THETA).simulate(
            counts,
            flags,
            backend=backend,
        )
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return (
        statistics.median(elapsed_ms),
        min(elapsed_ms),
        max(elapsed_ms),
        elapsed_ms,
        trace,
        event_count,
    )


def _verify_rust_safety() -> dict[str, Any]:
    """Compile and execute the actual standalone Rust-safety tests."""
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/mcculloch_pitts.rs"
    display_command = (
        "rustc --edition=2021 --test "
        f"{source.relative_to(REPOSITORY).as_posix()} -O -o <temp-binary> && <temp-binary>"
    )
    with tempfile.TemporaryDirectory(prefix="sc_neurocore_mcp_safety_") as temp_dir:
        binary = Path(temp_dir) / "mcculloch_pitts_safety_tests"
        command = ["rustc", "--edition=2021", "--test", str(source), "-O", "-o", str(binary)]
        try:
            compiled = subprocess.run(
                command,
                cwd=REPOSITORY,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            executed = (
                subprocess.run(
                    [str(binary)],
                    cwd=REPOSITORY,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if compiled.returncode == 0
                else compiled
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "command": display_command,
                "passed": False,
                "returncode": -1,
                "output_tail": [str(exc)],
            }
    output = (executed.stdout + "\n" + executed.stderr).strip().splitlines()
    return {
        "command": display_command,
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
    """Run the controlled benchmark and write its evidence artifact."""
    parser = argparse.ArgumentParser(
        description="Controlled McCulloch-Pitts five-backend benchmark"
    )
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
    reference: npt.NDArray[np.uint8] | None = None
    reference_ms: float | None = None
    reference_count: int | None = None
    parity_ok = True
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            rows[backend] = {"available": False, "used": False, "unavailable_reason": reason}
            continue
        median_ms, minimum_ms, maximum_ms, samples_ms, trace, event_count = _measure_backend(
            backend
        )
        if backend == "python":
            reference = trace
            reference_ms = median_ms
            reference_count = event_count
            mismatch_count = 0
        else:
            if reference is None or reference_ms is None or reference_count is None:
                raise RuntimeError("Python reference must be measured first")
            mismatch_count = int(np.count_nonzero(trace != reference))
            parity_ok &= mismatch_count == 0 and event_count == reference_count
        rows[backend] = {
            "available": True,
            "used": True,
            "median_call_ms": median_ms,
            "minimum_call_ms": minimum_ms,
            "maximum_call_ms": maximum_ms,
            "samples_call_ms": samples_ms,
            "median_ns_per_row": median_ms * 1_000_000.0 / N_ROWS,
            "speedup_vs_python": reference_ms / median_ms if reference_ms is not None else 1.0,
            "trace_mismatch_count": mismatch_count,
            "trace_matches_python": mismatch_count == 0,
            "event_count": event_count,
            "event_count_matches_python": (
                True if reference_count is None else event_count == reference_count
            ),
            "trace_sha256": _trace_digest(trace),
        }

    measured_order = sorted(
        (backend for backend in BACKENDS if rows[backend].get("used") is True),
        key=lambda backend: float(rows[backend]["median_call_ms"]),
    )
    native_order = [backend for backend in measured_order if backend != "python"]
    auto_order = [*native_order, "python"]
    rust_safety = _verify_rust_safety()
    counts, flags = _inputs(N_ROWS)
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "kernel": KERNEL,
        "evidence_class": "local_regression_single_cpu_affinity_non_exclusive",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": (
            "taskset --cpu-list <cpu> env PYTHONPATH=bridge:src:. .venv/bin/python "
            "benchmarks/bench_model_mcculloch_pitts.py --json <artifact>"
        ),
        "workload": {
            "n_rows": N_ROWS,
            "repeats": N_REPEATS,
            "warmup_rows": WARMUP_ROWS,
            "theta": THETA,
            "input_pattern": (
                "count=index mod 16 with final int32 max; inhibition on index mod 11 zero"
            ),
            **_input_digests(counts, flags),
        },
        "backends": rows,
        "measured_order": measured_order,
        "fastest_measured_native_backend": native_order[0] if native_order else None,
        "recommended_auto_backend": native_order[0] if native_order else "python",
        "auto_backend_order": auto_order,
        "auto_backend_selection_basis": (
            "same-host measured warm batch order; non-exclusive timings remain diagnostic"
        ),
        "binary_hashes": _binary_hashes(),
        "source_hashes": _source_hashes(),
        "verification": {"rust_safety": rust_safety},
        "meta": _environment(load_start),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not parity_ok:
        print("Five-backend McCulloch-Pitts parity failed")
        return 3
    if not rust_safety["passed"]:
        print("Standalone Rust-safety gate failed")
        return 5
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
