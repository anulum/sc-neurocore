#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""SC-NEUROCORE Python vs Rust Benchmark Suite.

Compares every Rust-accelerated hot path against its Python fallback,
plus end-to-end pipeline benchmarks.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np

sys.path.insert(0, "src")

# ── Benchmark infrastructure ────────────────────────────────────────


@dataclass
class BenchResult:
    name: str
    python_ms: float
    rust_ms: float
    speedup: float
    n: int = 0

    def row_rust(self) -> str:
        return (
            f"| {self.name:<48} | {self.n:>8} | "
            f"{self.python_ms:>10.2f} | {self.rust_ms:>10.2f} | "
            f"**{self.speedup:>7.1f}×** |"
        )

    def row_py(self) -> str:
        return (
            f"| {self.name:<48} | {self.n:>8} | {self.python_ms:>10.2f} | {'—':>10} | {'—':>10} |"
        )


def bench(fn, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[len(times) // 2]


results: List[BenchResult] = []

# ── Check Rust engine availability ──────────────────────────────────

try:
    from sc_neurocore_engine import (
        py_evo_batch_mutate,
        py_evo_batch_crossover,
        py_evo_diversity,
        py_evo_tournament,
    )
    import sc_neurocore_engine as _sne

    HAS_RUST = all(
        hasattr(_sne, _name)
        for _name in (
            "py_opt_sa_search",
            "py_opt_extract_pareto",
            "py_evo_batch_fitness",
            "py_evo_novelty",
            "py_ph_analyze_crosstalk",
            "py_ph_route_waveguides",
        )
    )
    del _sne
    print("✓ Rust engine (sc_neurocore_engine) loaded\n")
except ImportError as e:
    HAS_RUST = False
    print(f"✗ Rust engine not available: {e}\n")


# ═══════════════════════════════════════════════════════════════════
# 1. SA OPTIMIZER — Python vs Rust
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("1. Simulated Annealing Optimizer")
print("=" * 70)

from sc_neurocore.optimizer.sc_optimizer import (
    SCOptimizer,
    HardwareBudget,
    LayerProfile,
)

budget = HardwareBudget(max_luts=500_000, max_power_mw=5000.0)
opt = SCOptimizer(budget)

for n_layers, max_iter in [(5, 500), (20, 1000), (50, 2000)]:
    network = [
        LayerProfile(id=f"L{i}", mac_count=100 + i * 10, is_critical_path=(i == 0))
        for i in range(n_layers)
    ]

    # Force Python path
    py_ms = bench(
        lambda: opt._optimize_annealing_python(network, max_iter=max_iter, seed=42),
        warmup=1,
        repeats=3,
    )

    # Rust path
    if HAS_RUST:
        rust_ms = bench(
            lambda: opt._optimize_annealing_rust(network, max_iter=max_iter, seed=42),
            warmup=1,
            repeats=3,
        )
    else:
        rust_ms = py_ms

    speedup = py_ms / max(rust_ms, 0.001)
    r = BenchResult(
        f"SA Optimizer ({n_layers}L × {max_iter} iter)", py_ms, rust_ms, speedup, n_layers
    )
    results.append(r)
    tag = f"→ {speedup:.1f}×" if HAS_RUST else "(no Rust)"
    print(
        f"  {n_layers:>3}L × {max_iter:>4} iter:  Py={py_ms:>8.2f}ms  Rs={rust_ms:>8.2f}ms  {tag}"
    )


# ═══════════════════════════════════════════════════════════════════
# 2. EVOLUTIONARY OPERATIONS — Python vs Rust
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2. Evolutionary Operations")
print("=" * 70)

if HAS_RUST:
    for pop_size in [50, 200, 1000]:
        genome_len = 100
        rng = np.random.default_rng(42)
        pop = rng.standard_normal((pop_size, genome_len)).tolist()
        fitness = rng.random(pop_size).tolist()

        # ── Batch mutate ──
        def py_mutate():
            out = []
            for g in pop:
                m = [w + rng.normal(0, 0.1) if rng.random() < 0.5 else w for w in g]
                out.append(m)
            return out

        py_ms = bench(py_mutate, warmup=1, repeats=3)
        rust_ms = bench(lambda: py_evo_batch_mutate(pop, 0.5, 0.1, 42), warmup=1, repeats=3)
        speedup = py_ms / max(rust_ms, 0.001)
        results.append(
            BenchResult(
                f"Batch Mutate (pop={pop_size}×{genome_len})", py_ms, rust_ms, speedup, pop_size
            )
        )
        print(
            f"  Mutate  pop={pop_size:>5}: Py={py_ms:>8.2f}ms  Rs={rust_ms:>8.2f}ms  → {speedup:.1f}×"
        )

        # ── Tournament select ──
        def py_tournament():
            selected = []
            for _ in range(pop_size):
                idx = rng.choice(len(fitness), size=3, replace=False)
                best_idx = max(idx, key=lambda i: fitness[i])
                selected.append(best_idx)
            return selected

        py_ms = bench(py_tournament, warmup=1, repeats=3)
        rust_ms = bench(lambda: py_evo_tournament(fitness, pop_size, 3, 42), warmup=1, repeats=3)
        speedup = py_ms / max(rust_ms, 0.001)
        results.append(
            BenchResult(
                f"Tournament Select (pop={pop_size}, k=3)", py_ms, rust_ms, speedup, pop_size
            )
        )
        print(
            f"  Tourney pop={pop_size:>5}: Py={py_ms:>8.2f}ms  Rs={rust_ms:>8.2f}ms  → {speedup:.1f}×"
        )

        # ── Population diversity ──
        n_div = min(50, pop_size)

        def py_diversity():
            total = 0.0
            for i in range(n_div):
                for j in range(i + 1, n_div):
                    total += sum((a - b) ** 2 for a, b in zip(pop[i], pop[j]))
            return total

        py_ms = bench(py_diversity, warmup=1, repeats=3)
        rust_ms = bench(lambda: py_evo_diversity(pop[:n_div]), warmup=1, repeats=3)
        speedup = py_ms / max(rust_ms, 0.001)
        results.append(
            BenchResult(
                f"Population Diversity (n={n_div}×{genome_len})", py_ms, rust_ms, speedup, n_div
            )
        )
        print(f"  Divers. n={n_div:>5}: Py={py_ms:>8.2f}ms  Rs={rust_ms:>8.2f}ms  → {speedup:.1f}×")

        # ── Batch crossover ──
        pop_a = pop[: pop_size // 2]
        pop_b = pop[pop_size // 2 :]
        min_len = min(len(pop_a), len(pop_b))
        pop_a = pop_a[:min_len]
        pop_b = pop_b[:min_len]

        def py_crossover():
            children = []
            for a, b in zip(pop_a, pop_b):
                cx = len(a) // 2
                children.append(a[:cx] + b[cx:])
            return children

        py_ms = bench(py_crossover, warmup=1, repeats=3)
        rust_ms = bench(lambda: py_evo_batch_crossover(pop_a, pop_b, 42), warmup=1, repeats=3)
        speedup = py_ms / max(rust_ms, 0.001)
        results.append(
            BenchResult(
                f"Batch Crossover (n={min_len}×{genome_len})", py_ms, rust_ms, speedup, min_len
            )
        )
        print(
            f"  Crossov n={min_len:>5}: Py={py_ms:>8.2f}ms  Rs={rust_ms:>8.2f}ms  → {speedup:.1f}×"
        )

        print()
else:
    print("  [SKIP] Rust evo engine not available\n")


# ═══════════════════════════════════════════════════════════════════
# 3. NAS SEARCH — End-to-End
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("3. NAS Search (end-to-end)")
print("=" * 70)

from sc_neurocore.nas.sc_nas_engine import run_nas

for pop, gens in [(20, 10), (50, 25), (100, 50)]:
    ms = bench(
        lambda: run_nas(population_size=pop, num_generations=gens, seed=42), warmup=1, repeats=3
    )
    results.append(BenchResult(f"NAS Search (pop={pop}, gens={gens})", ms, ms, 1.0, pop * gens))
    print(f"  pop={pop:>3} × gens={gens:>3} = {pop * gens:>5} evals:  {ms:>8.2f}ms")


# ═══════════════════════════════════════════════════════════════════
# 4. PHOTONIC / FDTD
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4. Photonic Simulation")
print("=" * 70)

from sc_neurocore.optics.photonic_emitter import (
    FDTDSolver,
    FDTD2DSolver,
    CrosstalkModel,
    PhotonicCompiler,
)

# FDTD 1D
for grid in [500, 1000, 2000]:
    solver = FDTDSolver(grid_size=grid)
    solver.inject_pulse(grid // 4, 1550.0)
    ms = bench(lambda: solver.step(100), warmup=1, repeats=3)
    results.append(BenchResult(f"1D FDTD ({grid} cells × 100 steps)", ms, ms, 1.0, grid))
    print(f"  1D FDTD {grid:>5} cells:  {ms:>8.2f}ms")

# FDTD 2D
for nx, ny in [(100, 50), (200, 100)]:
    s2d = FDTD2DSolver(nx=nx, ny=ny)
    s2d.set_waveguide(ny // 2, 5)
    s2d.inject_source(10, ny // 2)
    ms = bench(lambda: s2d.step(50), warmup=1, repeats=3)
    results.append(BenchResult(f"2D FDTD ({nx}×{ny} × 50 steps)", ms, ms, 1.0, nx * ny))
    print(f"  2D FDTD {nx}×{ny:>3}:  {ms:>8.2f}ms")

# Photonic compiler (bitstream → netlist + FDTD)
bs = np.random.default_rng(42).integers(0, 2, size=256).astype(np.float64)
compiler = PhotonicCompiler()
ms = bench(
    lambda: compiler.compile_bitstream(bs, run_fdtd=True, fdtd_steps=50), warmup=1, repeats=3
)
results.append(BenchResult("Photonic Compile (256-bit + FDTD)", ms, ms, 1.0, 256))
print(f"  Compile 256-bit:  {ms:>8.2f}ms")

# Crosstalk analysis (Python path: ≤10 WGs, Rust path: >10 WGs)
for n_wg in [10, 50, 200]:
    model = CrosstalkModel()
    ms = bench(lambda: model.analyze_bank(n_wg, 200.0, 10.0), warmup=1, repeats=5)
    backend = "Rust" if HAS_RUST and n_wg > 10 else "Python"
    results.append(BenchResult(f"Crosstalk ({n_wg} WGs, {backend})", ms, ms, 1.0, n_wg))
    print(f"  Crosstalk {n_wg:>3} WGs ({backend}):  {ms:>8.4f}ms")


# ═══════════════════════════════════════════════════════════════════
# 5. ADAPTIVE CONTROLLER
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5. Closed-Loop Adaptive Controller")
print("=" * 70)

from sc_neurocore.core.types import HardwareBudget as CoreBudget, LayerSpec
from sc_neurocore.control.adaptive_loop import AdaptiveController, AdaptiveLoopConfig

budget_c = CoreBudget(max_luts=500_000, max_power_mw=5000.0)
layer_specs = [LayerSpec(layer_id="L0", neurons=32, mac_count=32)]
cfg = AdaptiveLoopConfig(drift_threshold=0.05, reoptimize_cooldown_s=0.0, sa_max_iter=100)
pattern = np.random.default_rng(42).integers(0, 2, size=256).astype(np.float64)

for n_steps in [50, 200, 500]:

    def run_loop():
        ctrl = AdaptiveController(budget_c, layer_specs, cfg)
        for _ in range(n_steps):
            ctrl.step(pattern, pattern)
        return len(ctrl.adaptation_log)

    ms = bench(run_loop, warmup=1, repeats=3)
    results.append(BenchResult(f"Adaptive Loop ({n_steps} steps)", ms, ms, 1.0, n_steps))
    print(f"  {n_steps:>3} steps:  {ms:>8.2f}ms")


# ═══════════════════════════════════════════════════════════════════
# 6. ENERGY REPORTER
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("6. Energy / Carbon / Thermal Reporter")
print("=" * 70)

from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter

reporter = UnifiedEnergyReporter(asic_power_mw=100.0)
ms = bench(lambda: reporter.analyze(total_power_mw=500.0), warmup=2, repeats=10)
results.append(BenchResult("Unified Energy Report", ms, ms, 1.0, 1))
print(f"  Single report:  {ms:>8.4f}ms")


# ═══════════════════════════════════════════════════════════════════
# 7. QUANTUM ANNEALING — Python vs Rust
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("7. Quantum Annealing (Ising SA)")
print("=" * 70)

from sc_neurocore.bridges.quantum_annealing import (
    SCToIsing,
    SimulatedAnnealer,
)

qa_compiler = SCToIsing()

for n_q in [20, 50, 100]:
    adj = np.random.default_rng(42).standard_normal((n_q, n_q))
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    qa_model = qa_compiler.compile(adj)
    sa_solver = SimulatedAnnealer(n_sweeps=500, seed=42)

    # Python path
    py_ms = bench(lambda: sa_solver._solve_ising_python(qa_model, num_reads=3), warmup=1, repeats=3)

    # Rust path (if available)
    if HAS_RUST and n_q > 10:
        rust_ms = bench(
            lambda: sa_solver._solve_ising_rust(qa_model, num_reads=3), warmup=1, repeats=3
        )
    else:
        rust_ms = py_ms

    speedup = py_ms / max(rust_ms, 0.001)
    results.append(
        BenchResult(f"QA Ising SA ({n_q}Q × 500sw × 3 reads)", py_ms, rust_ms, speedup, n_q)
    )
    tag = f"→ {speedup:.1f}×" if rust_ms != py_ms else "(Python only)"
    print(f"  {n_q:>3}Q: Py={py_ms:>8.1f}ms  Rs={rust_ms:>8.1f}ms  {tag}")


# ═══════════════════════════════════════════════════════════════════
# 8. END-TO-END PIPELINE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("7. End-to-End Pipeline")
print("=" * 70)

from sc_neurocore.nas.sc_nas_engine import NASVerilogEmitter


def e2e_small():
    r = run_nas(population_size=10, num_generations=5, seed=42)
    best = max(r.pareto_front, key=lambda c: c.accuracy)
    net = [LayerProfile(id=f"L{i}", mac_count=l.neurons) for i, l in enumerate(best.layers)]
    opt2 = SCOptimizer(budget)
    opt2.optimize_annealing(net, max_iter=100, seed=42)
    NASVerilogEmitter.emit(best)
    reporter.analyze(total_power_mw=best.total_power_mw)


def e2e_medium():
    r = run_nas(population_size=30, num_generations=15, seed=42)
    best = max(r.pareto_front, key=lambda c: c.accuracy)
    net = [LayerProfile(id=f"L{i}", mac_count=l.neurons) for i, l in enumerate(best.layers)]
    opt2 = SCOptimizer(budget)
    opt2.optimize_annealing(net, max_iter=500, seed=42)
    NASVerilogEmitter.emit(best)
    reporter.analyze(total_power_mw=best.total_power_mw)


def e2e_large():
    r = run_nas(population_size=50, num_generations=25, seed=42)
    best = max(r.pareto_front, key=lambda c: c.accuracy)
    net = [LayerProfile(id=f"L{i}", mac_count=l.neurons) for i, l in enumerate(best.layers)]
    opt2 = SCOptimizer(budget)
    opt2.optimize_annealing(net, max_iter=1000, seed=42)
    NASVerilogEmitter.emit(best)
    reporter.analyze(total_power_mw=best.total_power_mw)


for name, fn in [
    ("E2E Small (10×5)", e2e_small),
    ("E2E Medium (30×15)", e2e_medium),
    ("E2E Large (50×25)", e2e_large),
]:
    ms = bench(fn, warmup=1, repeats=3)
    results.append(BenchResult(f"Pipeline: {name}", ms, ms, 1.0, 1))
    print(f"  {name}:  {ms:>8.2f}ms")


# ═══════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════

print("\n\n")
print("╔" + "═" * 100 + "╗")
print("║" + " SC-NEUROCORE BENCHMARK RESULTS — Python vs Rust".center(100) + "║")
print("╚" + "═" * 100 + "╝")
print()

rust_benchmarks = [r for r in results if r.rust_ms != r.python_ms]
py_benchmarks = [r for r in results if r.rust_ms == r.python_ms]

hdr = f"| {'Benchmark':<48} | {'N':>8} | {'Python ms':>10} | {'Rust ms':>10} | {'Speedup':>10} |"
sep = f"|{'-' * 50}|{'-' * 10}|{'-' * 12}|{'-' * 12}|{'-' * 12}|"

print(hdr)
print(sep)

if rust_benchmarks:
    print(f"| {'🦀 Rust-Accelerated Hot Paths':<48} | {'':>8} | {'':>10} | {'':>10} | {'':>10} |")
    print(sep)
    for r in rust_benchmarks:
        print(r.row_rust())
    print(sep)

if py_benchmarks:
    print(
        f"| {'🐍 Python-Only (baselines & E2E)':<48} | {'':>8} | {'':>10} | {'':>10} | {'':>10} |"
    )
    print(sep)
    for r in py_benchmarks:
        print(r.row_py())

print(sep)

# Summary stats
if rust_benchmarks:
    avg_speedup = np.mean([r.speedup for r in rust_benchmarks])
    max_speedup = max(rust_benchmarks, key=lambda r: r.speedup)
    print(f"\n  Mean Rust speedup: {avg_speedup:.1f}×")
    print(f"  Peak Rust speedup: {max_speedup.speedup:.1f}× ({max_speedup.name})")

# Save JSON
out = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "rust_available": HAS_RUST,
    "benchmarks": [
        {
            "name": r.name,
            "n": r.n,
            "python_ms": round(r.python_ms, 3),
            "rust_ms": round(r.rust_ms, 3),
            "speedup": round(r.speedup, 1),
            "has_rust": r.rust_ms != r.python_ms,
        }
        for r in results
    ],
}
os.makedirs("benchmarks/results", exist_ok=True)
with open("benchmarks/results/py_vs_rust_integration.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n  Results saved → benchmarks/results/py_vs_rust_integration.json")
