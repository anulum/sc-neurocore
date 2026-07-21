#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled EscapeRate five-backend benchmark

"""Measure the canonical seeded EscapeRate batch through every public dispatcher.

The evidence binds real exact-RC/hazard timings to full trace parity, exact event
counts, exact final LFSR16 state, source hashes, CPU affinity, runtime versions,
and an executable Rust-safety check. Missing backends or unpinned execution fail
unless explicitly acknowledged.
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
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import escape_rate as backends
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_STEPS = 200_000
N_REPEATS = 7
WARMUP_STEPS = 1_000
CURRENT = 17.0
TRACE_ATOL = 2.0e-14
KERNEL = "escape_rate_exact_rc_hazard_lfsr16_batch"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
SOURCE_PATHS = (
    "benchmarks/bench_model_escape_rate.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/escape_rate.rs",
    "engine/src/lib.rs",
    "engine/src/neurons/trivial/escape_rate.rs",
    "engine/src/pyo3_neurons.rs",
    "src/sc_neurocore/accel/escape_rate.py",
    "src/sc_neurocore/accel/go/neurons/escape_rate/escape_rate.go",
    "src/sc_neurocore/accel/go/neurons/escape_rate/libescape_rate.h",
    "src/sc_neurocore/accel/go/services/escape_rate.go",
    "src/sc_neurocore/accel/go/services/escape_rate_test.go",
    "src/sc_neurocore/accel/julia/neurons/escape_rate.jl",
    "src/sc_neurocore/accel/mojo/kernels/escape_rate.mojo",
    "src/sc_neurocore/accel/rust/safety/escape_rate.rs",
    "src/sc_neurocore/neurons/_stochastic_threshold.py",
    "src/sc_neurocore/neurons/model_schemas/escape_rate.json",
    "src/sc_neurocore/neurons/model_schemas/escape_rate.toml",
    "src/sc_neurocore/neurons/models/escape_rate.py",
)


def _configured() -> EscapeRateNeuron:
    """Return a non-default cell exercising the complete seeded native ABI."""
    return EscapeRateNeuron(
        v=-64.0,
        v_rest=-68.0,
        v_reset=-66.0,
        v_threshold=-52.0,
        tau_m=12.5,
        rho_0=0.02,
        delta_u=4.0,
        resistance=1.3,
        dt=0.25,
        seed=0x1234,
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


def _trace_digest(trace: npt.NDArray[np.float64]) -> str:
    """Return a byte-order-stable SHA-256 digest for one measured trace."""
    return hashlib.sha256(np.asarray(trace, dtype="<f8").tobytes()).hexdigest()


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Return backend availability plus a deterministic diagnostic."""
    if backend == "python":
        return True, ""
    if backend == "rust":
        return backends._HAS_RUST, "" if backends._HAS_RUST else "Rust engine batch unavailable"
    if backend == "julia":
        available = backends.ensure_julia_loaded()
        return available, "" if available else "juliacall or EscapeRate module unavailable"
    if backend == "go":
        available = backends.ensure_go_loaded()
        return available, "" if available else "compiled Go libescape_rate.so unavailable"
    available = backends.ensure_mojo_loaded()
    return available, "" if available else "compiled Mojo libescape_rate.so unavailable"


def _measure_backend(
    backend: str,
) -> tuple[
    float,
    float,
    float,
    list[float],
    npt.NDArray[np.float64],
    int,
    tuple[float, int],
]:
    """Warm one public batch lane, then return timings and final state."""
    _configured().simulate(min(WARMUP_STEPS, N_STEPS), CURRENT, backend=backend)
    elapsed_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    spikes = 0
    final_state = (-64.0, 0x1234)
    for _repeat in range(N_REPEATS):
        gc.collect()
        neuron = _configured()
        started = time.perf_counter_ns()
        trace, spikes = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        final_state = (neuron.v, neuron.rng_state)
    return (
        statistics.median(elapsed_ms),
        min(elapsed_ms),
        max(elapsed_ms),
        elapsed_ms,
        trace,
        spikes,
        final_state,
    )


def _verify_rust_safety() -> dict[str, Any]:
    """Compile and execute the actual standalone Rust-safety test module."""
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/escape_rate.rs"
    display_command = (
        "rustc --edition=2021 --test "
        f"{source.relative_to(REPOSITORY).as_posix()} -O -o <temp-binary> && <temp-binary>"
    )
    with tempfile.TemporaryDirectory(prefix="sc_neurocore_escape_rate_safety_") as temp_dir:
        binary = Path(temp_dir) / "escape_rate_safety_tests"
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
    parser = argparse.ArgumentParser(description="Controlled EscapeRate five-backend benchmark")
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
    reference_state: tuple[float, int] | None = None
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            rows[backend] = {"available": False, "used": False, "unavailable_reason": reason}
            continue
        median_ms, minimum_ms, maximum_ms, samples_ms, trace, spikes, final_state = (
            _measure_backend(backend)
        )
        if backend == "python":
            reference = trace
            reference_ms = median_ms
            reference_spikes = spikes
            reference_state = final_state
            trace_parity = 0.0
            state_parity = 0.0
        else:
            if (
                reference is None
                or reference_ms is None
                or reference_spikes is None
                or reference_state is None
            ):
                raise RuntimeError("Python reference must be measured first")
            trace_parity = float(np.max(np.abs(trace - reference))) if trace.size else 0.0
            state_parity = abs(final_state[0] - reference_state[0])
        rows[backend] = {
            "available": True,
            "used": True,
            "median_call_ms": median_ms,
            "minimum_call_ms": minimum_ms,
            "maximum_call_ms": maximum_ms,
            "samples_call_ms": samples_ms,
            "median_ns_per_step": median_ms * 1_000_000.0 / N_STEPS,
            "speedup_vs_python": reference_ms / median_ms if reference_ms is not None else 1.0,
            "trace_max_abs_diff": trace_parity,
            "state_max_abs_diff": state_parity,
            "parity_max_abs_diff": max(trace_parity, state_parity),
            "event_count": spikes,
            "event_count_matches_python": (
                True if reference_spikes is None else spikes == reference_spikes
            ),
            "rng_state_matches_python": (
                True if reference_state is None else final_state[1] == reference_state[1]
            ),
            "trace_sha256": _trace_digest(trace),
            "final_state": {"v": final_state[0], "rng_state": final_state[1]},
        }

    measured_order = sorted(
        (backend for backend in BACKENDS if rows[backend].get("used") is True),
        key=lambda backend: float(rows[backend]["median_call_ms"]),
    )
    native_order = [backend for backend in measured_order if backend != "python"]
    rust_safety = _verify_rust_safety()
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "kernel": KERNEL,
        "evidence_class": "local_regression_single_cpu_affinity_non_exclusive",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": (
            "taskset --cpu-list <cpu> env PYTHONPATH=src:. .venv/bin/python "
            "benchmarks/bench_model_escape_rate.py --json <artifact>"
        ),
        "workload": {
            "n_steps": N_STEPS,
            "repeats": N_REPEATS,
            "warmup_steps": WARMUP_STEPS,
            "current": CURRENT,
            "initial_state": {"v": -64.0, "rng_state": 0x1234},
            "parameters": "complete non-default 9-double plus seeded LFSR16 native ABI",
            "trace_atol": TRACE_ATOL,
        },
        "meta": _environment(load_start),
        "backends": rows,
        "measured_order": measured_order,
        "recommended_auto_backend": native_order[0] if native_order else "python",
        "verification": {"rust_safety": rust_safety},
        "source_hashes": _source_hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"EscapeRate benchmark: {N_STEPS} steps x {N_REPEATS} repeats")
    for backend in measured_order:
        row = rows[backend]
        print(
            f"{backend:>7}: {float(row['median_call_ms']):10.3f} ms  "
            f"{float(row['speedup_vs_python']):8.2f}x  "
            f"max|delta|={float(row['parity_max_abs_diff']):.3e}  "
            f"events={int(row['event_count'])}  rng={int(row['final_state']['rng_state'])}"
        )
    print(f"Measured order: {', '.join(measured_order)}")
    print(f"Recommended auto backend: {report['recommended_auto_backend']}")
    print(f"Rust safety tests: {'PASS' if rust_safety['passed'] else 'FAIL'}")
    print(f"Wrote {args.json}")

    if not rust_safety["passed"]:
        return 5
    if any(
        not bool(row.get("event_count_matches_python", True))
        or not bool(row.get("rng_state_matches_python", True))
        for row in rows.values()
    ):
        return 3
    if any(
        float(row.get("parity_max_abs_diff", 0.0)) > TRACE_ATOL
        for row in rows.values()
        if row.get("used") is True
    ):
        return 4
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
