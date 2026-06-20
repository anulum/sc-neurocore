#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark for the public SC inference surface (sc_forward)

"""Measure and compare the Rust and NumPy backends of ``sc_forward``.

The input encoder is the deterministic 16-bit LFSR comparator, so the Rust path
and the NumPy fallback are bit-identical for a fixed seed; the benchmark asserts
that exact parity and records per-backend throughput plus host-load and CPU
affinity context per the benchmark-core-isolation policy.

    taskset -c 10-11 python benchmarks/bench_sc_forward.py \\
        --json benchmarks/results/bench_sc_forward.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.accel import available_backends, get_backend  # noqa: E402
from sc_neurocore.accel.sc_inference import _lfsr_encode_bits  # noqa: E402
from sc_neurocore.accel.vector_ops import pack_bitstream  # noqa: E402


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _cpuset() -> str | None:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return line.split(":", 1)[1].strip()
    return _read_text(Path("/sys/fs/cgroup/cpuset.cpus.effective"))


def _cpu_model() -> str:
    info = _read_text(Path("/proc/cpuinfo"))
    if info is not None:
        for line in info.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _host_context(load_before: list[float], load_after: list[float]) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = _cpuset()
    shielded = cpuset == "10-11" or affinity == [10, 11]
    return {
        "affinity_cpus": affinity,
        "cgroup_effective_cpuset": cpuset,
        "load_average_before": load_before,
        "load_average_after": load_after,
        "runtime_cpuset_shield_claimed": shielded,
        "isolation_mode": "runtime-cpuset-shield" if shielded else "non-isolated-shared-host",
    }


def _pack_weights(
    weights: npt.NDArray[np.float64], length: int, seed: int
) -> npt.NDArray[np.uint64]:
    n_out, n_in = weights.shape
    bits = _lfsr_encode_bits(weights.reshape(-1), length, seed)
    packed = np.stack([pack_bitstream(bits[k]) for k in range(n_out * n_in)])
    return packed.reshape(n_out, n_in, -1).astype(np.uint64)


def _time(
    handle: Any,
    weights: npt.NDArray[np.uint64],
    probs: npt.NDArray[np.float64],
    length: int,
    repeats: int,
) -> tuple[float, npt.NDArray[np.float64]]:
    result = handle.sc_forward(weights, probs, length=length)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = handle.sc_forward(weights, probs, length=length)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2], result


def run(n_out: int, n_in: int, length: int, seed: int, repeats: int) -> dict[str, Any]:
    """Run the Rust and NumPy backends and return the benchmark report."""
    rng = np.random.default_rng(seed)
    weights = rng.random((n_out, n_in))
    probs = rng.random(n_in)
    packed = _pack_weights(weights, length, seed=0x55AA)
    macs = n_out * n_in * length
    availability = available_backends()
    load_before = list(os.getloadavg())

    backends: dict[str, dict[str, Any]] = {}
    results: dict[str, npt.NDArray[np.float64]] = {}
    for name in ("numpy", "rust"):
        if not availability.get(name, False):
            backends[name] = {"available": False, "used": False, "reason": "backend not present"}
            continue
        runs = 3 if name == "numpy" else repeats
        wall, result = _time(get_backend(name), packed, probs, length, runs)
        results[name] = result
        backends[name] = {
            "available": True,
            "used": True,
            "median_call_ms": round(wall * 1e3, 6),
            "mac_per_s": round(macs / wall, 1),
            "repeats": runs,
        }

    if (
        "numpy" in backends
        and backends["numpy"].get("used")
        and "rust" in backends
        and backends["rust"].get("used")
    ):
        backends["rust"]["speedup_over_numpy"] = round(
            backends["numpy"]["median_call_ms"] / backends["rust"]["median_call_ms"], 2
        )

    parity: dict[str, Any] = {"reference": "numpy"}
    accuracy_vs_dense = float(np.abs(results["numpy"] - weights @ probs).max())
    if "rust" in results:
        delta = int(np.abs(results["rust"].view(np.int64) - results["numpy"].view(np.int64)).max())
        parity["rust_numpy_bit_identical"] = bool(np.array_equal(results["rust"], results["numpy"]))
        parity["rust_numpy_raw_delta"] = delta

    load_after = list(os.getloadavg())
    return {
        "benchmark": "sc_forward_packed",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language_surfaces": ["Python", "Rust"],
        "kernel": "sc_forward",
        "hardware_measurement_claimed": False,
        "workload": {
            "n_out": n_out,
            "n_in": n_in,
            "length": length,
            "multiply_accumulates": macs,
            "seed": seed,
        },
        "accuracy_max_abs_error_vs_dense": accuracy_vs_dense,
        "meta": {
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "backends": backends,
        "parity": parity,
        "host_context": _host_context(load_before, load_after),
    }


def main() -> int:
    """Parse arguments, run the benchmark and persist the JSON artefact."""
    parser = argparse.ArgumentParser(description="Benchmark sc_forward (Rust vs NumPy).")
    parser.add_argument("--outputs", type=int, default=128)
    parser.add_argument("--inputs", type=int, default=128)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument(
        "--json", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "bench_sc_forward.json"
    )
    args = parser.parse_args()
    report = run(args.outputs, args.inputs, args.length, args.seed, args.repeats)

    print(f"{'Backend':<8}{'MAC/s':>18}{'Call (ms)':>14}{'Speedup':>10}")
    print("-" * 50)
    for name in ("numpy", "rust"):
        info = report["backends"].get(name, {})
        if not info.get("used"):
            print(f"{name:<8}{'MISSING':>18}")
            continue
        speed = info.get("speedup_over_numpy", 1.0)
        print(f"{name:<8}{info['mac_per_s']:>18,.0f}{info['median_call_ms']:>14.4f}{speed:>9.2f}×")
    parity = report["parity"]
    print(f"\nRust↔NumPy bit-identical: {parity.get('rust_numpy_bit_identical', 'n/a')}")
    print(f"Max abs error vs dense W@probs: {report['accuracy_max_abs_error_vs_dense']:.4f}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Results → {args.json.relative_to(REPO_ROOT)}")
    return 0 if parity.get("rust_numpy_bit_identical", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
