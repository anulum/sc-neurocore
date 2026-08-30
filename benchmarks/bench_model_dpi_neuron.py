# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled DPI five-backend benchmark

"""Measure the published coupled DPI recurrence through public dispatch.

The evidence record binds public-API timings to source hashes, CPU affinity,
host load, runtime versions, exact event parity, bounded state parity, final
state, and executable Rust-safety verification. Missing backends or unpinned
execution fail unless explicitly acknowledged.
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

from sc_neurocore.accel import dpi_neuron as backends
from sc_neurocore.neurons.models.dpi_neuron import DPINeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_STEPS = 100_000
N_REPEATS = 7
CURRENT = 5.0
STATE_ATOL = 5.0e-13
KERNEL = "dpi_neuron_indiveri_2010_eq2_eq3_euler_constant_current"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
DISPATCH_ORDER = ("go", "julia", "mojo", "rust", "python")
SOURCE_PATHS = (
    "benchmarks/bench_model_dpi_neuron.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/hardware/dpi.rs",
    "engine/src/neurons/hardware/dpi_neuron.rs",
    "src/sc_neurocore/accel/dpi_neuron.py",
    "src/sc_neurocore/accel/go/neurons/dpi_neuron/dpi_neuron.go",
    "src/sc_neurocore/accel/go/neurons/dpi_neuron/libdpi_neuron.h",
    "src/sc_neurocore/accel/go/services/dpi_neuron.go",
    "src/sc_neurocore/accel/go/services/dpi_neuron_test.go",
    "src/sc_neurocore/accel/julia/dpi_neuron_parity_test.jl",
    "src/sc_neurocore/accel/julia/neurons/dpi_neuron.jl",
    "src/sc_neurocore/accel/mojo/kernels/dpi_neuron.mojo",
    "src/sc_neurocore/accel/rust/examples/dpi_neuron_trace.rs",
    "src/sc_neurocore/accel/rust/safety/dpi_neuron.rs",
    "src/sc_neurocore/neurons/models/dpi_neuron.py",
    "src/sc_neurocore/neurons/reference_receipts/dpi_indiveri_stefanini_chicca_2010.json",
    "src/sc_neurocore/neurons/model_schemas/dpi_neuron.toml",
    "src/sc_neurocore/neurons/model_schemas/dpi_neuron.json",
    "tests/cosim_reference_dpi_neuron.py",
    "tests/test_cosim_dpi_neuron.py",
    "tests/test_dpi_neuron_backend_loading.py",
    "tests/test_dpi_neuron_backends_auto_dispatch.py",
    "tests/test_dpi_neuron_backends_backend_parity.py",
    "tests/test_dpi_neuron_backends_rejects_and_hints.py",
    "tests/test_model_dpi_neuron_contract_rejects.py",
    "tests/test_model_dpi_neuron_dynamics_and_golden.py",
    "tests/test_model_dpi_neuron_rust_and_reset.py",
    "tests/test_model_descriptor_reproducibility.py",
    "tests/test_reference_dpi_neuron.py",
    "tools/emit_catalogue_formal.py",
    "hdl/formal/catalogue/sc_dpineuron.sby",
    "hdl/formal/catalogue/sc_dpineuron.v",
    "hdl/formal/catalogue/sc_dpineuron_formal.v",
    "hdl/reports/yosys_dpi_neuron_q1616_2026-08-30.json",
    "src/sc_neurocore/neurons/model_descriptors/DPINeuron.toml",
)

_CompletePacket = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
]


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
        by_suffix = nested.setdefault(stem, {})
        assert isinstance(by_suffix, dict)
        by_suffix[suffix] = digest
    return {**flat, **nested}


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Return backend availability plus a deterministic diagnostic."""
    if backend == "python":
        return True, ""
    if backend == "rust":
        return (
            backends._HAS_RUST,
            "" if backends._HAS_RUST else "Rust engine DPI symbol unavailable",
        )
    if backend == "julia":
        available = backends.ensure_julia_loaded()
        return available, "" if available else "juliacall or DPI module unavailable"
    if backend == "go":
        available = backends.ensure_go_loaded()
        return available, "" if available else "compiled Go libdpi_neuron.so unavailable"
    available = backends.ensure_mojo_loaded()
    return available, "" if available else "compiled Mojo libdpi_neuron.so unavailable"


def _measure_backend(
    backend: str,
) -> tuple[
    float,
    float,
    _CompletePacket,
    tuple[float, float, float],
]:
    """Warm one backend, then return timings and final numerical state."""
    DPINeuron().simulate_complete(20, CURRENT, backend=backend)
    elapsed_ms: list[float] = []
    packet: _CompletePacket = (
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.uint8),
    )
    final_state = (0.0, 0.0, 0.0)
    for _repeat in range(N_REPEATS):
        gc.collect()
        neuron = DPINeuron()
        started = time.perf_counter_ns()
        packet = neuron.simulate_complete(N_STEPS, CURRENT, backend=backend)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        final_state = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    return statistics.median(elapsed_ms), min(elapsed_ms), packet, final_state


def _max_abs_diff(
    actual: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
) -> float:
    """Return the largest absolute state difference between two traces."""
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _verify_rust_safety() -> dict[str, Any]:
    """Execute the actual accel/rust/safety DPI test module."""
    command = [
        "cargo",
        "test",
        "--release",
        "--manifest-path",
        "src/sc_neurocore/accel/rust/Cargo.toml",
        "dpi_neuron::tests",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    repository_prefix = f"{REPOSITORY}/"
    output = [
        line.replace(repository_prefix, "")
        for line in (completed.stdout + "\n" + completed.stderr).strip().splitlines()
    ]
    return {
        "command": " ".join(command),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
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
    parser = argparse.ArgumentParser(description="Controlled DPI five-backend benchmark")
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
    reference: _CompletePacket | None = None
    reference_ms: float | None = None
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            rows[backend] = {"available": False, "used": False, "unavailable_reason": reason}
            continue
        median_ms, minimum_ms, packet, final_state = _measure_backend(backend)
        i_mem, i_ahp, refractory, events = packet
        if backend == "python":
            reference = packet
            reference_ms = median_ms
            parity_by_state = {name: 0.0 for name in ("i_mem", "i_ahp", "refractory_time")}
            event_trace_matches = True
        else:
            if reference is None or reference_ms is None:
                raise RuntimeError("Python reference must be measured first")
            parity_by_state = {
                name: _max_abs_diff(actual, expected)
                for name, actual, expected in zip(
                    ("i_mem", "i_ahp", "refractory_time"),
                    packet[:3],
                    reference[:3],
                    strict=True,
                )
            }
            event_trace_matches = bool(np.array_equal(events, reference[3]))
        parity = max(parity_by_state.values())
        spikes = int(np.sum(events, dtype=np.int64))
        reference_spikes = spikes if reference is packet else int(np.sum(reference[3]))
        rows[backend] = {
            "available": True,
            "used": True,
            "median_call_ms": median_ms,
            "minimum_call_ms": minimum_ms,
            "speedup_vs_python": reference_ms / median_ms if reference_ms is not None else 1.0,
            "parity_max_abs_diff": parity,
            "parity_max_abs_diff_by_state": parity_by_state,
            "event_count": spikes,
            "event_count_matches_python": spikes == reference_spikes,
            "event_trace_matches_python": event_trace_matches,
            "trace_sha256": {
                "i_mem_le_f64": hashlib.sha256(i_mem.astype("<f8").tobytes()).hexdigest(),
                "i_ahp_le_f64": hashlib.sha256(i_ahp.astype("<f8").tobytes()).hexdigest(),
                "refractory_time_le_f64": hashlib.sha256(
                    refractory.astype("<f8").tobytes()
                ).hexdigest(),
                "events_u8": hashlib.sha256(events.tobytes()).hexdigest(),
            },
            "final_state": dict(
                zip(("i_mem", "i_ahp", "refractory_time"), final_state, strict=True)
            ),
        }

    measured_order = sorted(
        (backend for backend in BACKENDS if rows[backend].get("used") is True),
        key=lambda backend: float(rows[backend]["median_call_ms"]),
    )
    rust_safety = _verify_rust_safety()
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "kernel": KERNEL,
        "workload": {
            "n_steps": N_STEPS,
            "repeats": N_REPEATS,
            "current": CURRENT,
            "parameters": (
                "DPI factory defaults; Indiveri-Stefanini-Chicca 2010 Eqs. (2)-(3); "
                "simultaneous explicit Euler; post-update threshold and refractory pulse"
            ),
            "state_atol": STATE_ATOL,
        },
        "meta": _environment(load_start),
        "backends": rows,
        "measured_order": measured_order,
        "dispatcher_order": list(DISPATCH_ORDER),
        "dispatcher_order_rationale": (
            "Go is probed before Julia to avoid Julia runtime initialisation when the "
            "shared Go ABI is available; Mojo and compatible Rust remain explicit fallbacks."
        ),
        "verification": {"rust_safety": rust_safety},
        "source_hashes": _source_hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"DPI benchmark: {N_STEPS} steps x {N_REPEATS} repeats")
    for backend in measured_order:
        row = rows[backend]
        print(
            f"{backend:>7}: {float(row['median_call_ms']):10.3f} ms  "
            f"{float(row['speedup_vs_python']):8.2f}x  "
            f"max abs delta={float(row['parity_max_abs_diff']):.3e}  "
            f"events={int(row['event_count'])}"
        )
    print(f"Measured order: {', '.join(measured_order)}")
    print(f"Rust safety tests: {'PASS' if rust_safety['passed'] else 'FAIL'}")
    print(f"Wrote {args.json}")

    if not rust_safety["passed"]:
        return 5
    if any(not bool(row.get("event_count_matches_python", True)) for row in rows.values()):
        return 3
    if any(not bool(row.get("event_trace_matches_python", True)) for row in rows.values()):
        return 3
    if any(
        float(row.get("parity_max_abs_diff", 0.0)) > STATE_ATOL
        for row in rows.values()
        if row.get("used") is True
    ):
        return 4
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
