# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic Doctor Benchmark (Python vs Rust)

"""Benchmark: Python vs Rust PyO3 for stochastic doctor operations.

Measures speedup across SCC, batch SCC, precision, and histogram at
multiple input sizes. Outputs JSON results to benchmarks/results/.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc_neurocore.stochastic_doctor.diagnostics import (
    _scc_python,
    StochasticDoctor,
    _HAS_PYO3,
    _sdc_rust,
)


def _bench(fn, *args, warmup: int = 3, repeats: int = 20) -> float:
    """Return median execution time in seconds."""
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def benchmark_scc(sizes: list[int]) -> list[dict]:
    """Benchmark SCC computation at varying stream lengths."""
    results = []
    rng = np.random.default_rng(42)
    for n in sizes:
        a = rng.integers(0, 2, size=n, dtype=np.uint8)
        b = rng.integers(0, 2, size=n, dtype=np.uint8)
        a_c = np.ascontiguousarray(a)
        b_c = np.ascontiguousarray(b)

        t_py = _bench(_scc_python, a_c, b_c)

        if _HAS_PYO3:
            t_rust = _bench(lambda: _sdc_rust.py_scc_bytes(a_c, b_c))
            speedup = t_py / t_rust if t_rust > 0 else float("inf")
        else:
            t_rust = None
            speedup = 1.0

        entry = {
            "function": "scc_bytes",
            "input_size": n,
            "python_seconds": round(t_py, 8),
            "rust_seconds": round(t_rust, 8) if t_rust else None,
            "speedup": f"{speedup:.1f}x",
        }
        results.append(entry)
        print(
            f"  SCC  N={n:>8d}  Python={t_py:.6f}s  Rust={t_rust:.6f}s  → {speedup:.1f}×"
            if t_rust
            else f"  SCC  N={n:>8d}  Python={t_py:.6f}s  (no Rust)"
        )
    return results


def benchmark_batch_scc(neuron_counts: list[int], stream_len: int = 2048) -> list[dict]:
    """Benchmark batch N×N SCC computation."""
    results = []
    rng = np.random.default_rng(42)
    doc = StochasticDoctor()

    for n in neuron_counts:
        streams = rng.integers(0, 2, size=(n, stream_len), dtype=np.uint8)
        streams_c = np.ascontiguousarray(streams)

        # Python path (force)
        def py_batch():
            for i in range(n):
                for j in range(i + 1, n):
                    _scc_python(streams_c[i], streams_c[j])

        t_py = _bench(py_batch, repeats=5)

        if _HAS_PYO3:
            t_rust = _bench(lambda: _sdc_rust.py_scc_batch(streams_c), repeats=10)
            speedup = t_py / t_rust if t_rust > 0 else float("inf")
        else:
            t_rust = None
            speedup = 1.0

        pairs = n * (n - 1) // 2
        entry = {
            "function": "scc_batch",
            "num_neurons": n,
            "stream_length": stream_len,
            "pairs": pairs,
            "python_seconds": round(t_py, 6),
            "rust_seconds": round(t_rust, 6) if t_rust else None,
            "speedup": f"{speedup:.1f}x",
        }
        results.append(entry)
        label = (
            f"  Batch  N={n:>3d} ({pairs:>5d} pairs)  Python={t_py:.4f}s  Rust={t_rust:.4f}s  → {speedup:.1f}×"
            if t_rust
            else f"  Batch  N={n:>3d}  Python={t_py:.4f}s"
        )
        print(label)
    return results


def benchmark_precision(sizes: list[int]) -> list[dict]:
    """Benchmark precision estimation."""
    results = []
    rng = np.random.default_rng(42)
    doc = StochasticDoctor()

    for n in sizes:
        bs = rng.integers(0, 2, size=n, dtype=np.uint8)
        bs_c = np.ascontiguousarray(bs)

        def py_prec():
            p = float(np.mean(bs_c))
            return (p, p * (1 - p) / n)

        t_py = _bench(py_prec)

        if _HAS_PYO3:
            t_rust = _bench(lambda: _sdc_rust.py_precision_bytes(bs_c))
            speedup = t_py / t_rust if t_rust > 0 else float("inf")
        else:
            t_rust = None
            speedup = 1.0

        entry = {
            "function": "precision_bytes",
            "input_size": n,
            "python_seconds": round(t_py, 8),
            "rust_seconds": round(t_rust, 8) if t_rust else None,
            "speedup": f"{speedup:.1f}x",
        }
        results.append(entry)
        print(
            f"  Prec  N={n:>8d}  Python={t_py:.6f}s  Rust={t_rust:.6f}s  → {speedup:.1f}×"
            if t_rust
            else f"  Prec  N={n:>8d}  Python={t_py:.6f}s"
        )
    return results


def main():
    print("=" * 70)
    print("SC-NEUROCORE  Stochastic Doctor — Python vs Rust Benchmark")
    print("=" * 70)
    print(f"PyO3 available: {_HAS_PYO3}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print()

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    batch_neurons = [4, 8, 16, 32, 64]

    print("--- SCC (single pair) ---")
    scc_results = benchmark_scc(sizes)

    print("\n--- Batch SCC (N×N matrix) ---")
    batch_results = benchmark_batch_scc(batch_neurons)

    print("\n--- Precision estimation ---")
    prec_results = benchmark_precision(sizes)

    report = {
        "module": "stochastic_doctor",
        "date": time.strftime("%Y-%m-%d"),
        "cpu": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "pyo3_available": _HAS_PYO3,
        "scc": scc_results,
        "batch_scc": batch_results,
        "precision": prec_results,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "stochastic_doctor_py_vs_rust.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
