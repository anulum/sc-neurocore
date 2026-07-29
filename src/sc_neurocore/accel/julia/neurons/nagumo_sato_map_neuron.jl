# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror of source-faithful Nagumo–Sato dynamics

"""Source-faithful one-state Nagumo–Sato refractory map."""
module NagumoSatoMapNeuronAccel

using PythonCall: PyArray

export NagumoSatoMapCandidateError,
    NagumoSatoMapConfigurationError,
    NagumoSatoMapNeuronState,
    is_candidate_error,
    is_configuration_error,
    output,
    reset!,
    simulate_nagumo_sato_map_b,
    simulate_nagumo_sato_map!,
    step!,
    valid

"""Invalid state, parameters, input, or buffer contract."""
struct NagumoSatoMapConfigurationError <: Exception
    message::String
end

"""Non-finite candidate produced by an otherwise valid step."""
struct NagumoSatoMapCandidateError <: Exception
    message::String
end

Base.showerror(io::IO, error::NagumoSatoMapConfigurationError) = print(io, error.message)
Base.showerror(io::IO, error::NagumoSatoMapCandidateError) = print(io, error.message)
is_configuration_error(error)::Bool = error isa NagumoSatoMapConfigurationError
is_candidate_error(error)::Bool = error isa NagumoSatoMapCandidateError

"""Mutable internal state and parameters for the reduced source map."""
mutable struct NagumoSatoMapNeuronState
    y::Float64
    k::Float64
    alpha::Float64
    bias::Float64
end

NagumoSatoMapNeuronState() = NagumoSatoMapNeuronState(0.1, 0.6, 1.0, 0.2)

"""Return whether state and parameters satisfy the source bounds."""
function valid(state::NagumoSatoMapNeuronState)::Bool
    return all(isfinite, (state.y, state.k, state.alpha, state.bias)) &&
        0.0 <= state.k < 1.0 && state.alpha > 0.0
end

"""Return the source unit-step output, using H(0)=1."""
output(state::NagumoSatoMapNeuronState)::Int = state.y >= 0.0 ? 1 : 0

"""Advance `y'=k*y-alpha*H(y)+bias+current` atomically."""
function step!(state::NagumoSatoMapNeuronState, current::Real = 0.0)::Int
    drive = Float64(current)
    isfinite(drive) && valid(state) ||
        throw(NagumoSatoMapConfigurationError("invalid Nagumo-Sato configuration or current"))
    next_y = state.k * state.y - state.alpha * output(state) + state.bias + drive
    isfinite(next_y) ||
        throw(NagumoSatoMapCandidateError("Nagumo-Sato candidate must be finite"))
    state.y = next_y
    return output(state)
end

"""Restore the source initial state while preserving parameters."""
function reset!(state::NagumoSatoMapNeuronState)::Nothing
    state.y = 0.1
    return nothing
end

_writable(buffer::AbstractVector{Float64}) =
    applicable(setindex!, buffer, 0.0, firstindex(buffer))
_writable(::PyArray{Float64, 1, W, C, R}) where {W, C, R} = W

function _validate_buffer(buffer::AbstractVector, name::String, steps::Int; writable::Bool = false)
    eltype(buffer) === Float64 ||
        throw(NagumoSatoMapConfigurationError("$name must have Float64 elements"))
    length(buffer) == steps ||
        throw(NagumoSatoMapConfigurationError("$name length mismatch"))
    (isempty(buffer) || (applicable(stride, buffer, 1) && stride(buffer, 1) == 1)) ||
        throw(NagumoSatoMapConfigurationError("$name must have unit stride"))
    if writable && !_writable(buffer)
        throw(NagumoSatoMapConfigurationError("$name must be writable"))
    end
    return nothing
end

"""Fill complete state/output/event buffers and return final receipts."""
function simulate_nagumo_sato_map!(
    y_init::Real,
    k::Real,
    alpha::Real,
    bias::Real,
    current::AbstractVector,
    y_out::AbstractVector,
    x_out::AbstractVector,
    spikes_out::AbstractVector,
)
    steps = length(current)
    _validate_buffer(current, "current", steps)
    _validate_buffer(y_out, "y_out", steps; writable = true)
    _validate_buffer(x_out, "x_out", steps; writable = true)
    _validate_buffer(spikes_out, "spikes_out", steps; writable = true)
    state = NagumoSatoMapNeuronState(Float64.((y_init, k, alpha, bias))...)
    valid(state) || throw(NagumoSatoMapConfigurationError("invalid Nagumo-Sato configuration"))
    all(isfinite, current) ||
        throw(NagumoSatoMapConfigurationError("current must contain only finite values"))
    y_trace = Vector{Float64}(undef, steps)
    x_trace = Vector{Float64}(undef, steps)
    spike_trace = Vector{Float64}(undef, steps)
    count = 0
    @inbounds for index in 1:steps
        event = step!(state, current[index])
        y_trace[index], x_trace[index], spike_trace[index] = state.y, event, event
        count += event
    end
    copyto!(y_out, y_trace)
    copyto!(x_out, x_trace)
    copyto!(spikes_out, spike_trace)
    return (state.y, output(state), count)
end

"""PythonCall-stable alias for the in-place batch function."""
simulate_nagumo_sato_map_b(args...) = simulate_nagumo_sato_map!(args...)

end # module NagumoSatoMapNeuronAccel
