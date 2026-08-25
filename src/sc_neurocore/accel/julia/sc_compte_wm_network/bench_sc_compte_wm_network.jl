# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Measured local-regression benchmark for the Julia SC Compte network lane."""

using Dates
using SHA
using Statistics
using TOML

include(joinpath(@__DIR__, "SCCompteWMNetwork.jl"))
using .SCCompteWMNetwork

const REPOSITORY = normpath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
const OUTPUT = joinpath(REPOSITORY, "benchmarks", "results", "bench_sc_compte_wm_network_julia.toml")
const SOURCE_PATHS = [
    "src/sc_neurocore/accel/julia/sc_compte_wm_network/Project.toml",
    "src/sc_neurocore/accel/julia/sc_compte_wm_network/Manifest.toml",
    "src/sc_neurocore/accel/julia/sc_compte_wm_network/SCCompteWMNetwork.jl",
    "src/sc_neurocore/accel/julia/sc_compte_wm_network/bench_sc_compte_wm_network.jl",
]

function file_sha256(path)
    open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

function build_payload(steps::Int=1000, repeats::Int=3)
    steps > 0 && repeats > 0 || throw(ArgumentError("steps and repeats must be positive"))
    run!(SCCompteWMNetworkRuntime(), 16 * DT_MS; statistics_window_ms=16 * DT_MS)
    samples_ns = Int[]
    input_digests = String[]
    spike_digests = String[]
    state_digests = String[]
    spike_counts = Tuple{Int,Int}[]
    for _ in 1:repeats
        runtime = SCCompteWMNetworkRuntime()
        started = time_ns()
        receipt = run!(runtime, steps * DT_MS; statistics_window_ms=500.0)
        push!(samples_ns, Int(time_ns() - started))
        push!(input_digests, receipt.input_sha256)
        push!(spike_digests, receipt.spike_sha256)
        push!(state_digests, receipt.final_state_sha256)
        push!(spike_counts, (receipt.excitatory_spikes, receipt.inhibitory_spikes))
    end
    deterministic = length(unique(input_digests)) == 1 &&
        length(unique(spike_digests)) == 1 && length(unique(state_digests)) == 1 &&
        length(unique(spike_counts)) == 1
    median_ns = round(Int, median(samples_ns))
    Dict(
        "schema_version" => "sc-neurocore.sc-compte-wm-network-benchmark.v1",
        "generated_at" => string(now(UTC)),
        "model" => "SC-COMPTE-WM-NETWORK",
        "execution_path" => "julia-midpoint-rk2-fftw",
        "evidence_class" => "local_regression",
        "production_speed_claimed" => false,
        "hardware_measurement_claimed" => false,
        "persistent_bump_claimed" => false,
        "distractor_resistance_claimed" => false,
        "configuration" => Dict(
            "cells" => 2560, "excitatory_cells" => N_EXCITATORY,
            "inhibitory_cells" => N_INHIBITORY, "dt_ms" => DT_MS,
            "steps" => steps, "duration_ms" => steps * DT_MS,
            "repeats" => repeats, "seed" => 42,
        ),
        "environment" => Dict(
            "julia" => string(VERSION), "kernel" => string(Sys.KERNEL),
            "architecture" => string(Sys.ARCH), "cpu_threads" => Sys.CPU_THREADS,
            "julia_threads" => Threads.nthreads(),
        ),
        "source_sha256" => Dict(path => file_sha256(joinpath(REPOSITORY, path)) for path in SOURCE_PATHS),
        "samples_ns" => samples_ns,
        "median_ns" => median_ns,
        "median_ns_per_network_step" => median_ns / steps,
        "median_cell_updates_per_second" => 2560 * steps / (median_ns / 1.0e9),
        "input_sha256" => input_digests[1],
        "spike_sha256" => spike_digests[1],
        "final_state_sha256" => state_digests[1],
        "spike_counts" => Dict("excitatory" => spike_counts[1][1],
                               "inhibitory" => spike_counts[1][2]),
        "repeat_receipts_exact" => deterministic,
        "passed" => deterministic,
    )
end

payload = build_payload()
open(OUTPUT, "w") do io
    TOML.print(io, payload; sorted=true)
end
println(TOML.print(payload; sorted=true))
payload["passed"] || exit(1)
