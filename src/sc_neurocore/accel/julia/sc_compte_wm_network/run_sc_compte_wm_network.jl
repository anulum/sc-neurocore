# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""JSON command adapter for public Julia SC Compte network dispatch."""

include(joinpath(@__DIR__, "SCCompteWMNetwork.jl"))
using .SCCompteWMNetwork

function parse_stimulus(encoded::String)
    fields = split(encoded, ',')
    length(fields) == 5 || error("stimulus must have start,duration,current,kind,center")
    kind = Symbol(fields[4])
    center = fields[5] == "none" ? nothing : parse(Float64, fields[5])
    SCCompteWMStimulus(parse(Float64, fields[1]), parse(Float64, fields[2]),
                       parse(Float64, fields[3]); kind=kind, center_deg=center)
end

function parse_options(args)
    duration_ms = nothing
    statistics_window_ms = nothing
    seed = UInt64(42)
    structured_ei = false
    modulated = false
    allow_recurrent_autapses = false
    stimuli = SCCompteWMStimulus[]
    index = 1
    while index <= length(args)
        flag = args[index]
        if flag in ("--duration-ms", "--statistics-window-ms", "--seed", "--stimulus")
            index < length(args) || error("missing value for $flag")
            value = args[index + 1]
            if flag == "--duration-ms"
                duration_ms = parse(Float64, value)
            elseif flag == "--statistics-window-ms"
                statistics_window_ms = parse(Float64, value)
            elseif flag == "--seed"
                seed = parse(UInt64, value)
            else
                push!(stimuli, parse_stimulus(value))
            end
            index += 2
        elseif flag == "--structured-ei"
            structured_ei = true
            index += 1
        elseif flag == "--modulated"
            modulated = true
            index += 1
        elseif flag == "--allow-recurrent-autapses"
            allow_recurrent_autapses = true
            index += 1
        else
            error("unknown argument: $flag")
        end
    end
    duration_ms === nothing && error("--duration-ms is required")
    statistics_window_ms === nothing && error("--statistics-window-ms is required")
    spec = SCCompteWMNetworkSpec(; seed, structured_ei, modulated,
                                  allow_recurrent_autapses)
    (; duration_ms, statistics_window_ms, spec, stimuli)
end

function print_statistics(statistics)
    if statistics === nothing
        print("null")
        return
    end
    print("{\"excitatory_rate_hz\":", statistics.excitatory_rate_hz,
          ",\"inhibitory_rate_hz\":", statistics.inhibitory_rate_hz,
          ",\"bump_angle_deg\":", statistics.bump_angle_deg,
          ",\"resultant_length\":", statistics.resultant_length,
          ",\"circular_width_deg\":")
    statistics.circular_width_deg === nothing ? print("null") :
        print(statistics.circular_width_deg)
    print('}')
end

function print_receipt(receipt, elapsed_ns)
    print("{\"runtime\":\"julia\",\"execution_ns\":", elapsed_ns,
          ",\"specification_version\":\"", receipt.specification_version,
          "\",\"seed\":", receipt.seed,
          ",\"duration_ms\":", receipt.duration_ms,
          ",\"steps\":", receipt.steps,
          ",\"excitatory_spikes\":", receipt.excitatory_spikes,
          ",\"inhibitory_spikes\":", receipt.inhibitory_spikes,
          ",\"input_sha256\":\"", receipt.input_sha256,
          "\",\"spike_sha256\":\"", receipt.spike_sha256,
          "\",\"final_state_sha256\":\"", receipt.final_state_sha256,
          "\",\"windows\":[")
    for (index, window) in enumerate(receipt.windows)
        index > 1 && print(',')
        print("{\"start_ms\":", window.start_ms,
              ",\"end_ms\":", window.end_ms,
              ",\"excitatory_spikes\":", window.excitatory_spikes,
              ",\"inhibitory_spikes\":", window.inhibitory_spikes,
              ",\"statistics\":")
        print_statistics(window.statistics)
        print('}')
    end
    println("]}")
end

options = parse_options(ARGS)
runtime = SCCompteWMNetworkRuntime(options.spec)
started = time_ns()
receipt = run!(runtime, options.duration_ms; stimuli=options.stimuli,
               statistics_window_ms=options.statistics_window_ms)
print_receipt(receipt, time_ns() - started)
