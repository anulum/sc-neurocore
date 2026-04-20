# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo reference implementation for evo_substrate hot paths
#
# Minimal, bit-exact reference of the three compute kernels (genomic
# distance, uniform crossover, point mutation) for multi-language parity
# + benchmarking vs Rust (crates/evo_substrate_core), Julia
# (accel/julia/evo_substrate/evo_substrate_bench.jl), Go
# (accel/go/evo_substrate/), Python
# (src/sc_neurocore/evo_substrate/evo_substrate.py).
#
# This file is strictly an *accel-chain reference* for the hot-path
# benchmark — not a port of the 40+ orchestration classes in the Python
# module. The Python orchestration (ReplicationEngine, LineageTracker,
# etc.) stays in Python; only the compute kernels are mirrored here.

from time import perf_counter_ns
from math import abs
from collections import List

alias EPSILON: Float64 = 1.0e-10


fn genomic_distance(a: List[Float64], b: List[Float64]) -> Float64:
    """Scale-invariant L1 distance: mean(|a-b|/(|a|+|b|+eps))."""
    var n = len(a)
    if n == 0:
        return 0.0
    var acc: Float64 = 0.0
    for i in range(n):
        var diff = abs(a[i] - b[i])
        var norm = abs(a[i]) + abs(b[i]) + EPSILON
        acc += diff / norm
    return acc / Float64(n)


fn crossover_uniform(
    a: List[Float64],
    b: List[Float64],
    mask: List[UInt8],
) -> List[Float64]:
    """Syswerda uniform crossover — picks a[i] when mask[i] != 0, else b[i]."""
    var n = len(a)
    var out = List[Float64]()
    for i in range(n):
        if mask[i] != 0:
            out.append(a[i])
        else:
            out.append(b[i])
    return out^


fn point_mutation(
    gene: List[Float64],
    mask: List[UInt8],
    noise: List[Float64],
) -> List[Float64]:
    """Gaussian multiplicative point mutation; caller supplies the noise."""
    var n = len(gene)
    var out = List[Float64]()
    for i in range(n):
        if mask[i] != 0:
            out.append(gene[i] + noise[i] * (abs(gene[i]) + 1.0e-8))
        else:
            out.append(gene[i])
    return out^


fn population_diversity(
    population: List[List[Float64]],
) -> Float64:
    """Mean pairwise genomic_distance across a population."""
    var n = len(population)
    if n < 2:
        return 0.0
    var acc: Float64 = 0.0
    var count: Float64 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            acc += genomic_distance(population[i], population[j])
            count += 1.0
    return acc / count


fn run_benchmark() raises:
    """Time the three kernels over a 19-D vector and print ns/call."""
    alias D: Int = 19
    alias ITERS: Int = 100_000

    var a = List[Float64]()
    var b = List[Float64]()
    var mask = List[UInt8]()
    var noise = List[Float64]()
    for i in range(D):
        a.append(Float64(i + 1) * 0.1)
        b.append(Float64(i + 1) * 0.2)
        mask.append(UInt8(i % 2))
        noise.append(0.01)

    # Warm-up
    _ = genomic_distance(a, b)
    _ = crossover_uniform(a, b, mask)
    _ = point_mutation(a, mask, noise)

    var sink: Float64 = 0.0

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        sink += genomic_distance(a, b)
    var dt_gd = Float64(perf_counter_ns() - t0) / Float64(ITERS)

    var t1 = perf_counter_ns()
    for _ in range(ITERS):
        var out = crossover_uniform(a, b, mask)
        sink += out[0]
    var dt_cr = Float64(perf_counter_ns() - t1) / Float64(ITERS)

    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        var out = point_mutation(a, mask, noise)
        sink += out[0]
    var dt_pm = Float64(perf_counter_ns() - t2) / Float64(ITERS)

    # Print so the value of `sink` cannot be constant-folded out.
    print("sink", sink)
    print("genomic_distance_ns_per_call ", dt_gd)
    print("crossover_uniform_ns_per_call ", dt_cr)
    print("point_mutation_ns_per_call ", dt_pm)


fn main() raises:
    run_benchmark()
