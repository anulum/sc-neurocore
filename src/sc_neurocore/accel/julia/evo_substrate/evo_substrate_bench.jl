# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia reference implementation for evo_substrate hot paths

"""
Minimal, bit-exact reference of the three compute kernels (genomic
distance, uniform crossover, point mutation) for multi-language parity
+ benchmarking vs Rust (crates/evo_substrate_core), Go
(accel/go/evo_substrate/), Mojo (accel/mojo/kernels/evo_substrate_bench.mojo),
Python (src/sc_neurocore/evo_substrate/evo_substrate.py).

This file is strictly an *accel-chain reference* for the hot-path
benchmark — not a port of the 40+ orchestration classes in the Python
module. The Python orchestration (ReplicationEngine, LineageTracker, etc.)
stays in Python; only the compute kernels are mirrored here.

Matches `sc_neurocore.evo_substrate.evo_substrate.genomic_distance` exactly:
`mean(abs(a - b) ./ (abs(a) + abs(b) + 1e-10))`.
"""

module EvoSubstrateBench

const EPSILON = 1e-10

"""Scale-invariant L1 distance between two real vectors."""
function genomic_distance(a::AbstractVector{<:Real},
                          b::AbstractVector{<:Real})
    @assert length(a) == length(b)
    if isempty(a)
        return 0.0
    end
    n = length(a)
    acc = 0.0
    @inbounds for i in 1:n
        diff = abs(a[i] - b[i])
        norm = abs(a[i]) + abs(b[i]) + EPSILON
        acc += diff / norm
    end
    return acc / n
end

"""Syswerda uniform crossover: picks a[i] when mask[i] != 0, else b[i]."""
function crossover_uniform(a::AbstractVector{<:Real},
                           b::AbstractVector{<:Real},
                           mask::AbstractVector{<:Integer})
    @assert length(a) == length(b) == length(mask)
    out = similar(a)
    @inbounds for i in 1:length(a)
        out[i] = mask[i] != 0 ? a[i] : b[i]
    end
    return out
end

"""
In-place Gaussian multiplicative point mutation. Caller supplies the
noise vector so the kernel stays pure/deterministic under a fixed seed
on the caller side.
"""
function point_mutation!(gene::AbstractVector{Float64},
                         mask::AbstractVector{<:Integer},
                         noise::AbstractVector{Float64})
    @assert length(gene) == length(mask) == length(noise)
    @inbounds for i in 1:length(gene)
        if mask[i] != 0
            gene[i] += noise[i] * (abs(gene[i]) + 1e-8)
        end
    end
    return gene
end

"""Mean pairwise distance across the rows of an `n × d` population matrix."""
function population_diversity(population::AbstractMatrix{<:Real})
    n = size(population, 1)
    if n < 2
        return 0.0
    end
    acc = 0.0
    count = 0.0
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            acc += genomic_distance(view(population, i, :),
                                    view(population, j, :))
            count += 1.0
        end
    end
    return acc / count
end

"""
Run a micro-benchmark over the three kernels; prints Julia wall times in
nanoseconds per call. Invoked from
`benchmarks/bench_evo_substrate_multilang.py` via a Julia subprocess.
"""
function run_benchmark(; iters::Int = 100_000, d::Int = 19)
    a = collect(1:d) .* 0.1
    b = collect(1:d) .* 0.2
    mask = UInt8.([i % 2 for i in 1:d])
    noise = fill(0.01, d)

    # Warm-up one iteration each so Julia's JIT doesn't skew the numbers.
    genomic_distance(a, b)
    crossover_uniform(a, b, mask)
    point_mutation!(copy(a), mask, noise)

    results = Dict{String,Float64}()

    t0 = time_ns()
    for _ in 1:iters
        genomic_distance(a, b)
    end
    results["genomic_distance_ns_per_call"] = (time_ns() - t0) / iters

    t0 = time_ns()
    for _ in 1:iters
        crossover_uniform(a, b, mask)
    end
    results["crossover_uniform_ns_per_call"] = (time_ns() - t0) / iters

    t0 = time_ns()
    for _ in 1:iters
        point_mutation!(copy(a), mask, noise)
    end
    results["point_mutation_ns_per_call"] = (time_ns() - t0) / iters

    for (k, v) in results
        println("$(k)  $(round(v, digits=1)) ns")
    end
    return results
end

end  # module EvoSubstrateBench

# Allow direct `julia evo_substrate_bench.jl` invocation to run the benchmark.
if abspath(PROGRAM_FILE) == @__FILE__
    EvoSubstrateBench.run_benchmark()
end
