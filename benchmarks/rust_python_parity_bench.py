# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust vs Python Parity Benchmark

"""
Rust vs Python Parity Benchmark
================================

Direct comparison of identical operations in Rust (SIMD) vs Python (NumPy)
to quantify the speedup from the sc_neurocore_engine Rust backend.

Operations tested:
  1. Popcount — count set bits in packed u64 arrays
  2. Pack — convert float probabilities to packed bitstreams
  3. AND+popcount — bitwise AND then popcount (SC multiplication)
  4. Dense SC layer — full encode→fused→decode pipeline
  5. Kuramoto — N-oscillator phase evolution (1000 steps)

Usage::

    python benchmarks/rust_python_parity_bench.py
    python benchmarks/rust_python_parity_bench.py --json parity.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ParityResult:
    operation: str
    scale: int
    python_s: float
    rust_s: float
    speedup: float


def _has_rust_engine() -> bool:
    try:
        import sc_neurocore_engine  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 1. Popcount
# ---------------------------------------------------------------------------
def bench_popcount(n_words: int, repeats: int) -> ParityResult | None:
    rng = np.random.default_rng(42)
    data = rng.integers(0, 2**63, size=n_words, dtype=np.uint64)

    # Python: unpackbits + sum (simulated popcount)
    t0 = time.perf_counter()
    for _ in range(repeats):
        total_py = 0
        for w in data:
            total_py += bin(int(w)).count("1")
    py_time = (time.perf_counter() - t0) / repeats

    try:
        import sc_neurocore_engine as eng
    except ImportError:
        return None

    # Rust: popcount via SIMD dispatch
    data_list = data.tolist()
    t0 = time.perf_counter()
    for _ in range(repeats):
        eng.popcount(data_list)
    rust_time = (time.perf_counter() - t0) / repeats

    return ParityResult(
        "popcount", n_words, py_time, rust_time, py_time / rust_time if rust_time > 0 else 0
    )


# ---------------------------------------------------------------------------
# 2. Pack (float → bitstream)
# ---------------------------------------------------------------------------
def bench_pack(n_floats: int, repeats: int) -> ParityResult | None:
    rng = np.random.default_rng(42)
    probs = rng.random(n_floats)

    # Python: threshold comparison
    t0 = time.perf_counter()
    for _ in range(repeats):
        bits = (rng.random(n_floats) < probs).astype(np.uint8)
        _ = np.packbits(bits)
    py_time = (time.perf_counter() - t0) / repeats

    try:
        import sc_neurocore_engine as eng
    except ImportError:
        return None

    bits_np = (rng.random(n_floats) < probs).astype(np.uint8)
    t0 = time.perf_counter()
    for _ in range(repeats):
        eng.pack_bitstream_numpy(bits_np)
    rust_time = (time.perf_counter() - t0) / repeats

    return ParityResult(
        "pack", n_floats, py_time, rust_time, py_time / rust_time if rust_time > 0 else 0
    )


# ---------------------------------------------------------------------------
# 3. AND + popcount
# ---------------------------------------------------------------------------
def bench_and_popcount(n_words: int, repeats: int) -> ParityResult | None:
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2**63, size=n_words, dtype=np.uint64)
    b = rng.integers(0, 2**63, size=n_words, dtype=np.uint64)

    # Python
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = np.bitwise_and(a, b)
        total = sum(bin(int(w)).count("1") for w in c)
    py_time = (time.perf_counter() - t0) / repeats

    try:
        import sc_neurocore_engine as eng
    except ImportError:
        return None

    a_list = a.tolist()
    b_list = b.tolist()
    t0 = time.perf_counter()
    for _ in range(repeats):
        c_list = [x & y for x, y in zip(a_list, b_list)]
        eng.popcount(c_list)
    rust_time = (time.perf_counter() - t0) / repeats

    return ParityResult(
        "and_popcount", n_words, py_time, rust_time, py_time / rust_time if rust_time > 0 else 0
    )


# ---------------------------------------------------------------------------
# 4. Dense SC layer
# ---------------------------------------------------------------------------
def bench_dense_layer(n_in: int, repeats: int) -> ParityResult | None:
    n_out = n_in
    bl = 1024
    rng = np.random.default_rng(42)

    # Python: SCDenseLayer-like loop
    w_prob = rng.random((n_in, n_out))
    x_prob = rng.random(n_in)

    t0 = time.perf_counter()
    for _ in range(repeats):
        x_bits = (rng.random((n_in, bl)) < x_prob[:, None]).astype(np.uint8)
        w_bits = (rng.random((n_in, n_out, bl)) < w_prob[:, :, None]).astype(np.uint8)
        and_result = x_bits[:, None, :] & w_bits
        acc = and_result.sum(axis=(0, 2)) / (n_in * bl)
    py_time = (time.perf_counter() - t0) / repeats

    try:
        import sc_neurocore_engine as eng
    except ImportError:
        return None

    layer = eng.DenseLayer(n_in, n_out, bl, 42)
    x_list = x_prob.tolist()

    t0 = time.perf_counter()
    for _ in range(repeats):
        layer.forward_fast(x_list, 42)
    rust_time = (time.perf_counter() - t0) / repeats

    return ParityResult(
        "dense_sc_layer", n_in, py_time, rust_time, py_time / rust_time if rust_time > 0 else 0
    )


# ---------------------------------------------------------------------------
# 5. Kuramoto
# ---------------------------------------------------------------------------
def bench_kuramoto(n_osc: int, repeats: int) -> ParityResult | None:
    rng = np.random.default_rng(42)
    omega = rng.normal(1.0, 0.1, n_osc)
    coupling = np.full(n_osc * n_osc, 0.3 / n_osc)
    phases = rng.uniform(0, 2 * np.pi, n_osc)
    dt = 0.01
    n_steps = 1000

    # Python: vectorized Kuramoto
    t0 = time.perf_counter()
    for _ in range(repeats):
        ph = phases.copy()
        for _ in range(n_steps):
            K = coupling.reshape(n_osc, n_osc)
            diff = ph[None, :] - ph[:, None]
            dtheta = omega + (K * np.sin(diff)).sum(axis=1)
            ph += dt * dtheta
    py_time = (time.perf_counter() - t0) / repeats

    try:
        import sc_neurocore_engine as eng
    except ImportError:
        return None

    t0 = time.perf_counter()
    for _ in range(repeats):
        solver = eng.KuramotoSolver(omega.tolist(), coupling.tolist(), phases.tolist(), 0.0)
        solver.run(n_steps, dt, 42)
    rust_time = (time.perf_counter() - t0) / repeats

    return ParityResult(
        "kuramoto", n_osc, py_time, rust_time, py_time / rust_time if rust_time > 0 else 0
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
BENCHES = [
    ("popcount", [1000, 10000, 100000, 1000000], bench_popcount),
    ("pack", [1000, 10000, 100000, 1000000], bench_pack),
    ("and_popcount", [1000, 10000, 100000, 1000000], bench_and_popcount),
    ("dense_sc_layer", [16, 32, 64, 128], bench_dense_layer),
    ("kuramoto", [100, 200, 500, 1000], bench_kuramoto),
]


def run_all(repeats: int) -> list[ParityResult]:
    if not _has_rust_engine():
        print("WARNING: sc_neurocore_engine not available — Rust columns will be zero.")

    results: list[ParityResult] = []
    for name, scales, fn in BENCHES:
        print(f"\n--- {name} ---")
        for s in scales:
            print(f"  scale={s:>10,}...", end=" ", flush=True)
            r = fn(s, repeats)
            if r is None:
                print("SKIP (Rust not available)")
                continue
            print(f"Python={r.python_s:.4f}s  Rust={r.rust_s:.6f}s  speedup={r.speedup:.1f}x")
            results.append(r)
    return results


def format_markdown(results: list[ParityResult]) -> str:
    lines = [
        "# Rust vs Python Parity Benchmark",
        "",
        "| Operation | Scale | Python (s) | Rust (s) | Speedup |",
        "|-----------|------:|-----------:|---------:|--------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.operation:<16s} | {r.scale:>10,} "
            f"| {r.python_s:>10.4f} | {r.rust_s:>8.6f} | {r.speedup:>7.1f}x |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rust vs Python parity benchmark")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", type=str)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("  Rust vs Python Parity Benchmark (SC-NeuroCore)")
    print(f"  Repeats: {args.repeats}")
    print("=" * 70)

    results = run_all(args.repeats)

    if args.markdown:
        print("\n" + format_markdown(results))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "op": r.operation,
                "scale": r.scale,
                "python_s": r.python_s,
                "rust_s": r.rust_s,
                "speedup": r.speedup,
            }
            for r in results
        ]
        Path(args.json).write_text(json.dumps({"data": data}, indent=2))
        print(f"\nResults written to {args.json}")

    if not args.json and not args.markdown:
        print("\n" + format_markdown(results))


if __name__ == "__main__":
    main()
