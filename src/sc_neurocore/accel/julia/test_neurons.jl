# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia neuron solver test harness

using Printf

function test_all_neurons(neurons_dir::String)
    files = filter(f -> endswith(f, ".jl"), readdir(neurons_dir))
    sort!(files)

    passed = 0
    failed = 0
    errors = String[]
    timings = Tuple{String, Float64}[]

    println("=" ^ 70)
    println("SC-NeuroCore Julia Neuron Solver Test Suite")
    println("=" ^ 70)
    println("Testing $(length(files)) neuron models...")
    println()

    for fname in files
        model_name = replace(fname, ".jl" => "")
        fpath = joinpath(neurons_dir, fname)

        try
            # Include the file to define the module
            include(fpath)

            # Get the module
            mod_name = Symbol(join(capitalize.(split(model_name, "_"))) * "Accel")
            mod = getfield(Main, mod_name)

            # Time the simulate function
            t0 = time()
            result = mod.simulate(1000; I_ext=10.0, dt=0.1)
            elapsed = (time() - t0) * 1000  # ms

            # Validate output
            if result isa Tuple
                trace, spikes = result
                valid = length(trace) == 1000 && !all(isnan, trace)
            else
                trace = result
                valid = length(trace) == 1000
                spikes = 0
            end

            if valid
                passed += 1
                push!(timings, (model_name, elapsed))
                @printf("  ✅ %-40s %8.3f ms  (spikes=%d)\n", model_name, elapsed, spikes)
            else
                failed += 1
                push!(errors, "$model_name: invalid output")
                @printf("  ❌ %-40s INVALID OUTPUT\n", model_name)
            end
        catch e
            failed += 1
            err_msg = sprint(showerror, e)
            # Truncate long error messages
            if length(err_msg) > 80
                err_msg = err_msg[1:80] * "..."
            end
            push!(errors, "$model_name: $err_msg")
            @printf("  ❌ %-40s %s\n", model_name, err_msg[1:min(40, length(err_msg))])
        end
    end

    # Summary
    println()
    println("=" ^ 70)
    println("RESULTS: $passed passed, $failed failed out of $(length(files))")
    println("=" ^ 70)

    if !isempty(timings)
        sort!(timings, by=x->x[2], rev=true)
        println("\nTop 10 slowest models:")
        for (name, t) in timings[1:min(10, length(timings))]
            @printf("  %-40s %8.3f ms\n", name, t)
        end

        avg_time = sum(t for (_, t) in timings) / length(timings)
        @printf("\nAverage simulation time: %.3f ms\n", avg_time)
    end

    if !isempty(errors)
        println("\nFailed models:")
        for err in errors
            println("  • $err")
        end
    end

    return passed, failed
end

# Helper
function capitalize(s::AbstractString)
    isempty(s) ? s : uppercase(s[1]) * s[2:end]
end

# Run
if abspath(PROGRAM_FILE) == @__FILE__
    neurons_dir = joinpath(@__DIR__, "neurons")
    passed, failed = test_all_neurons(neurons_dir)
    exit(failed > 0 ? 1 : 0)
end
