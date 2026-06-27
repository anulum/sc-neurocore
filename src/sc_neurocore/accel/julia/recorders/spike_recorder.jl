# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for recorders/spike_recorder

module SpikeRecorderAccel

export BitstreamSpikeRecorderState,
    as_array,
    firing_rate_hz,
    isi_intervals_ms,
    record!,
    reset!,
    total_spikes

mutable struct BitstreamSpikeRecorderState
    dt_ms::Float64
    spikes::Vector{UInt8}
end

function BitstreamSpikeRecorderState()
    BitstreamSpikeRecorderState(1.0, UInt8[])
end

function BitstreamSpikeRecorderState(dt_ms::Real)
    dt_ms < 0 && throw(ArgumentError("dt_ms must be non-negative."))
    BitstreamSpikeRecorderState(Float64(dt_ms), UInt8[])
end

function _validate_spike(spike::Integer)::UInt8
    (spike == 0 || spike == 1) || throw(ArgumentError("Spike must be 0 or 1."))
    UInt8(spike)
end

function record!(state::BitstreamSpikeRecorderState, spike::Integer)::Nothing
    push!(state.spikes, _validate_spike(spike))
    nothing
end

function reset!(state::BitstreamSpikeRecorderState)::Nothing
    empty!(state.spikes)
    nothing
end

function as_array(state::BitstreamSpikeRecorderState)::Vector{UInt8}
    copy(state.spikes)
end

function total_spikes(state::BitstreamSpikeRecorderState)::Int
    Int(sum(state.spikes))
end

function firing_rate_hz(state::BitstreamSpikeRecorderState)::Float64
    sample_count = length(state.spikes)
    if sample_count == 0
        return 0.0
    end

    duration_ms = sample_count * state.dt_ms
    if duration_ms == 0.0
        return 0.0
    end

    total_spikes(state) / (duration_ms / 1000.0)
end

function isi_intervals_ms(state::BitstreamSpikeRecorderState)::Vector{Float64}
    spike_indices = findall(==(UInt8(1)), state.spikes)
    if length(spike_indices) < 2
        return Float64[]
    end

    Float64.(diff(spike_indices)) .* state.dt_ms
end

end # module SpikeRecorderAccel
