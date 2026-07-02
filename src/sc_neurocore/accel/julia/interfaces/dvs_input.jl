# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia safety mirror for interfaces/dvs_input

module DvsInputAccel

using Random

export DVSEvent, DVSInputLayerState, generate_bitstream_frame, process_events, validate_dvs_input

const DEFAULT_RNG_SEED = UInt64(0x4456535f494e5054)

struct DVSEvent
    x::Int
    y::Int
    timestamp_ms::Float64
    polarity::Int
end

mutable struct DVSInputLayerState
    height::Int
    width::Int
    decay_tau::Float64
    surface::Matrix{Float64}
    last_update_time::Float64
    rng::MersenneTwister
end

function DVSInputLayerState(
    height::Integer=1,
    width::Integer=1,
    decay_tau::Real=100.0;
    seed::Integer=DEFAULT_RNG_SEED,
)::DVSInputLayerState
    height_i = _positive_integer(height, "height must be positive")
    width_i = _positive_integer(width, "width must be positive")
    tau = _positive_finite_float(decay_tau, "decay_tau must be finite and positive")
    seed_i = Int(mod(UInt64(seed), UInt64(typemax(UInt32))))
    rng = MersenneTwister(seed_i)
    return DVSInputLayerState(height_i, width_i, tau, zeros(Float64, height_i, width_i), 0.0, rng)
end

function process_events(s::DVSInputLayerState, events::AbstractVector{DVSEvent})::Matrix{Float64}
    isempty(events) && return _output_probabilities(s)
    _validate_events(s, events)

    current_time = events[end].timestamp_ms
    dt = current_time - s.last_update_time
    decay_factor = exp(-dt / s.decay_tau)
    isfinite(decay_factor) || throw(ArgumentError("event decay factor must remain finite"))

    s.surface .*= decay_factor
    for event in events
        if _contains(s, event.x, event.y)
            s.surface[event.y + 1, event.x + 1] += 1.0
        end
    end
    s.last_update_time = current_time
    return _output_probabilities(s)
end

function generate_bitstream_frame(s::DVSInputLayerState, length::Integer)::Array{UInt8,3}
    length_i = _positive_integer(length, "length must be a positive integer")
    probs = _output_probabilities(s)
    bits = Array{UInt8}(undef, s.height, s.width, length_i)
    for y in 1:s.height, x in 1:s.width, idx in 1:length_i
        bits[y, x, idx] = rand(s.rng) < probs[y, x] ? UInt8(1) : UInt8(0)
    end
    return bits
end

function validate_dvs_input(s::DVSInputLayerState)::Bool
    return s.height > 0 &&
        s.width > 0 &&
        isfinite(s.decay_tau) &&
        s.decay_tau > 0.0 &&
        isfinite(s.last_update_time) &&
        s.last_update_time >= 0.0 &&
        size(s.surface) == (s.height, s.width) &&
        all(isfinite, s.surface)
end

function _positive_integer(value::Integer, message::AbstractString)::Int
    value > 0 || throw(ArgumentError(String(message)))
    return Int(value)
end

function _positive_integer(value::Bool, message::AbstractString)::Int
    throw(ArgumentError(String(message)))
end

function _positive_finite_float(value::Real, message::AbstractString)::Float64
    scalar = Float64(value)
    isfinite(scalar) && scalar > 0.0 || throw(ArgumentError(String(message)))
    return scalar
end

function _positive_finite_float(value::Bool, message::AbstractString)::Float64
    throw(ArgumentError(String(message)))
end

function _validate_events(s::DVSInputLayerState, events::AbstractVector{DVSEvent})::Nothing
    previous_time = nothing
    for event in events
        isfinite(event.timestamp_ms) || throw(ArgumentError("event timestamp must be finite"))
        if previous_time !== nothing && event.timestamp_ms < previous_time
            throw(ArgumentError("event timestamps must be monotonically non-decreasing"))
        end
        if event.timestamp_ms < s.last_update_time
            throw(ArgumentError("event timestamp cannot be earlier than last update time"))
        end
        event.polarity in (-1, 0, 1) || throw(ArgumentError("event polarity must be -1, 0, or 1"))
        previous_time = event.timestamp_ms
    end
    return nothing
end

function _contains(s::DVSInputLayerState, x::Integer, y::Integer)::Bool
    return x >= 0 && y >= 0 && x < s.width && y < s.height
end

function _output_probabilities(s::DVSInputLayerState)::Matrix{Float64}
    return tanh.(s.surface)
end

end # module DvsInputAccel
