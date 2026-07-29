# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror of source-faithful Aihara dynamics

module AiharaMapNeuronAccel

using PythonCall: PyArray

export AiharaMapCandidateError,
    AiharaMapConfigurationError,
    AiharaMapNeuronState,
    is_candidate_error,
    is_configuration_error,
    logistic,
    reset!,
    simulate_aihara_map_b,
    simulate_aihara_map!,
    step!,
    valid

struct AiharaMapConfigurationError <: Exception
    message::String
end

struct AiharaMapCandidateError <: Exception
    message::String
end

Base.showerror(io::IO, error::AiharaMapConfigurationError) = print(io, error.message)
Base.showerror(io::IO, error::AiharaMapCandidateError) = print(io, error.message)
is_configuration_error(error)::Bool = error isa AiharaMapConfigurationError
is_candidate_error(error)::Bool = error isa AiharaMapCandidateError

mutable struct AiharaMapNeuronState
    y::Float64
    k::Float64
    alpha::Float64
    bias::Float64
    epsilon::Float64
end

AiharaMapNeuronState() = AiharaMapNeuronState(0.1, 0.7, 1.0, 0.3968, 0.01)

function valid(state::AiharaMapNeuronState)::Bool
    return all(isfinite, (state.y, state.k, state.alpha, state.bias, state.epsilon)) &&
        0.0 <= state.k < 1.0 && state.alpha > 0.0 && state.epsilon > 0.0
end

function logistic(value::Float64, epsilon::Float64)::Float64
    argument = value / epsilon
    if argument >= 0.0
        return 1.0 / (1.0 + exp(-argument))
    end
    exponential = exp(argument)
    return exponential / (1.0 + exponential)
end

function step!(state::AiharaMapNeuronState, current::Real = 0.0)::Int
    drive = Float64(current)
    isfinite(drive) && valid(state) ||
        throw(AiharaMapConfigurationError("invalid Aihara state, parameters, or current"))
    next_y = state.k * state.y - state.alpha * logistic(state.y, state.epsilon) +
        state.bias + drive
    isfinite(next_y) ||
        throw(AiharaMapCandidateError("Aihara map candidate must be finite"))
    state.y = next_y
    return logistic(next_y, state.epsilon) >= 0.5 ? 1 : 0
end

function reset!(state::AiharaMapNeuronState)::Nothing
    state.y = 0.1
    return nothing
end

_writable(buffer::AbstractVector{Float64}) =
    applicable(setindex!, buffer, 0.0, firstindex(buffer))
_writable(::PyArray{Float64, 1, W, C, R}) where {W, C, R} = W

function _validate_buffer(buffer::AbstractVector, name::String, steps::Int; writable::Bool = false)
    eltype(buffer) === Float64 ||
        throw(AiharaMapConfigurationError("$name must have Float64 elements"))
    length(buffer) == steps ||
        throw(AiharaMapConfigurationError("$name length mismatch"))
    (isempty(buffer) || (applicable(stride, buffer, 1) && stride(buffer, 1) == 1)) ||
        throw(AiharaMapConfigurationError("$name must have unit stride"))
    if writable && !_writable(buffer)
        throw(AiharaMapConfigurationError("$name must be writable"))
    end
    return nothing
end

function simulate_aihara_map!(
    y_init::Real,
    k::Real,
    alpha::Real,
    bias::Real,
    epsilon::Real,
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
    state = AiharaMapNeuronState(Float64.((y_init, k, alpha, bias, epsilon))...)
    valid(state) || throw(AiharaMapConfigurationError("invalid Aihara configuration"))
    all(isfinite, current) ||
        throw(AiharaMapConfigurationError("current must contain only finite values"))

    y_trace = Vector{Float64}(undef, steps)
    x_trace = Vector{Float64}(undef, steps)
    spike_trace = Vector{Float64}(undef, steps)
    count = 0
    @inbounds for index in 1:steps
        event = step!(state, current[index])
        y_trace[index] = state.y
        x_trace[index] = logistic(state.y, state.epsilon)
        spike_trace[index] = Float64(event)
        count += event
    end
    copyto!(y_out, y_trace)
    copyto!(x_out, x_trace)
    copyto!(spikes_out, spike_trace)
    return (state.y, logistic(state.y, state.epsilon), count)
end

simulate_aihara_map_b(args...) = simulate_aihara_map!(args...)

end # module AiharaMapNeuronAccel
