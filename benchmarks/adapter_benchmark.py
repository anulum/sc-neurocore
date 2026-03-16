# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark L1-L16 holonomic adapters: latency, memory,

"""Benchmark L1-L16 holonomic adapters: latency, memory, throughput."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np


def benchmark_adapter(adapter_cls, n_steps=100):
    """Benchmark a single adapter instance."""
    tracemalloc.start()
    try:
        adapter = adapter_cls()
    except Exception:
        return None

    # Warm up
    try:
        dummy = np.zeros(16)
        if hasattr(adapter, "step_jax"):
            adapter.step_jax(0.001, dummy)
        elif hasattr(adapter, "step"):
            adapter.step(0.001)
    except Exception:
        pass

    # Timed run
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter_ns()
        try:
            if hasattr(adapter, "step_jax"):
                adapter.step_jax(0.001)
            elif hasattr(adapter, "step"):
                adapter.step(0.001)
        except Exception:
            break
        times.append(time.perf_counter_ns() - t0)

    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if not times:
        return None

    return {
        "mean_latency_us": np.mean(times) / 1000,
        "p99_latency_us": np.percentile(times, 99) / 1000,
        "peak_memory_kb": peak_kb / 1024,
        "throughput_steps_per_sec": 1e9 / np.mean(times) if np.mean(times) > 0 else 0,
        "n_steps": len(times),
    }


def run_all_benchmarks(n_steps=100, output_dir="benchmarks/results"):
    from sc_neurocore.adapters.holonomic import _ADAPTERS

    results = {}
    for name, cls in _ADAPTERS.items():
        result = benchmark_adapter(cls, n_steps)
        if result:
            results[name] = result

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "adapter_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    lines = [
        "# Adapter Benchmark Results\n",
        "| Adapter | Latency (µs) | Memory (KB) | Throughput (steps/s) |",
        "|---------|-------------|-------------|---------------------|",
    ]
    for name, r in sorted(results.items()):
        lines.append(
            f"| {name} | {r['mean_latency_us']:.1f} | {r['peak_memory_kb']:.1f} | {r['throughput_steps_per_sec']:.0f} |"
        )
    with open(out / "adapter_benchmark.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
