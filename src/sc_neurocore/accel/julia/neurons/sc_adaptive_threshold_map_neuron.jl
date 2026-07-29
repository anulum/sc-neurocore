# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror of the retained SC adaptive-threshold map

"""Two-state SC-NeuroCore sigmoid map with an adaptive threshold."""
module SCAdaptiveThresholdMapNeuronAccel

using PythonCall: PyArray

export SCAdaptiveThresholdMapCandidateError,
    SCAdaptiveThresholdMapConfigurationError,
    SCAdaptiveThresholdMapNeuronState,
    is_candidate_error,
    is_configuration_error,
    reset!,
    simulate_sc_adaptive_threshold_map_b,
    simulate_sc_adaptive_threshold_map!,
    step!,
    valid

"""Invalid project-model state, parameters, input, or buffer contract."""
struct SCAdaptiveThresholdMapConfigurationError <: Exception
    message::String
end

"""Non-finite candidate produced by an otherwise valid step."""
struct SCAdaptiveThresholdMapCandidateError <: Exception
    message::String
end

Base.showerror(io::IO, error::SCAdaptiveThresholdMapConfigurationError) = print(io, error.message)
Base.showerror(io::IO, error::SCAdaptiveThresholdMapCandidateError) = print(io, error.message)
is_configuration_error(error)::Bool = error isa SCAdaptiveThresholdMapConfigurationError
is_candidate_error(error)::Bool = error isa SCAdaptiveThresholdMapCandidateError

"""Mutable two-state project model and its bounded parameters."""
mutable struct SCAdaptiveThresholdMapNeuronState
    x::Float64
    theta::Float64
    k::Float64
    beta::Float64
    gamma::Float64
    theta_spike::Float64
    x_threshold::Float64
end

SCAdaptiveThresholdMapNeuronState() =
    SCAdaptiveThresholdMapNeuronState(0.0, 0.0, 1.5, 0.95, 0.3, 0.8, 0.8)

"""Return whether state and parameters satisfy the project bounds."""
function valid(state::SCAdaptiveThresholdMapNeuronState)::Bool
    return isfinite(state.x) && -5.0 <= state.x <= 5.0 &&
        isfinite(state.theta) && -5.0 <= state.theta <= 5.0 &&
        isfinite(state.k) && 0.0 <= state.k <= 5.0 &&
        isfinite(state.beta) && 0.0 <= state.beta <= 1.0 &&
        isfinite(state.gamma) && 0.0 <= state.gamma <= 2.0 &&
        isfinite(state.theta_spike) && 0.0 <= state.theta_spike <= 2.0 &&
        isfinite(state.x_threshold) && 0.0 <= state.x_threshold <= 2.0
end

function _sigmoid(value::Float64)::Float64
    if value >= 0.0
        return 1.0 / (1.0 + exp(-value))
    end
    exponential = exp(value)
    return exponential / (1.0 + exponential)
end

"""Advance the simultaneous project recurrence atomically."""
function step!(state::SCAdaptiveThresholdMapNeuronState, current::Real = 0.0)::Int
    drive = Float64(current)
    isfinite(drive) && valid(state) || throw(
        SCAdaptiveThresholdMapConfigurationError("invalid SC adaptive-map configuration or current"),
    )
    previous_x = state.x
    activation = _sigmoid((state.x - state.theta) * 4.0)
    next_x = -state.x + state.k * activation + drive
    fired = state.x >= state.theta_spike ? 1.0 : 0.0
    next_theta = state.beta * state.theta + state.gamma * fired
    isfinite(next_x) && isfinite(next_theta) || throw(
        SCAdaptiveThresholdMapCandidateError("SC adaptive-map candidate must be finite"),
    )
    state.x = clamp(next_x, -5.0, 5.0)
    state.theta = clamp(next_theta, -5.0, 5.0)
    return state.x >= state.x_threshold && previous_x < state.x_threshold ? 1 : 0
end

"""Restore both project-model states while preserving parameters."""
function reset!(state::SCAdaptiveThresholdMapNeuronState)::Nothing
    state.x, state.theta = 0.0, 0.0
    return nothing
end

_writable(buffer::AbstractVector{Float64}) =
    applicable(setindex!, buffer, 0.0, firstindex(buffer))
_writable(::PyArray{Float64, 1, W, C, R}) where {W, C, R} = W

function _validate_buffer(buffer::AbstractVector, name::String, steps::Int; writable::Bool = false)
    eltype(buffer) === Float64 ||
        throw(SCAdaptiveThresholdMapConfigurationError("$name must have Float64 elements"))
    length(buffer) == steps ||
        throw(SCAdaptiveThresholdMapConfigurationError("$name length mismatch"))
    (isempty(buffer) || (applicable(stride, buffer, 1) && stride(buffer, 1) == 1)) ||
        throw(SCAdaptiveThresholdMapConfigurationError("$name must have unit stride"))
    if writable && !_writable(buffer)
        throw(SCAdaptiveThresholdMapConfigurationError("$name must be writable"))
    end
    return nothing
end

"""Fill complete state/event buffers and return final receipts."""
function simulate_sc_adaptive_threshold_map!(
    x_init::Real,
    theta_init::Real,
    k::Real,
    beta::Real,
    gamma::Real,
    theta_spike::Real,
    x_threshold::Real,
    current::AbstractVector,
    x_out::AbstractVector,
    theta_out::AbstractVector,
    spikes_out::AbstractVector,
)
    steps = length(current)
    _validate_buffer(current, "current", steps)
    _validate_buffer(x_out, "x_out", steps; writable = true)
    _validate_buffer(theta_out, "theta_out", steps; writable = true)
    _validate_buffer(spikes_out, "spikes_out", steps; writable = true)
    state = SCAdaptiveThresholdMapNeuronState(
        Float64.((x_init, theta_init, k, beta, gamma, theta_spike, x_threshold))...,
    )
    valid(state) ||
        throw(SCAdaptiveThresholdMapConfigurationError("invalid SC adaptive-map configuration"))
    all(isfinite, current) || throw(
        SCAdaptiveThresholdMapConfigurationError("current must contain only finite values"),
    )
    x_trace = Vector{Float64}(undef, steps)
    theta_trace = Vector{Float64}(undef, steps)
    spike_trace = Vector{Float64}(undef, steps)
    count = 0
    @inbounds for index in 1:steps
        event = step!(state, current[index])
        x_trace[index], theta_trace[index], spike_trace[index] = state.x, state.theta, event
        count += event
    end
    copyto!(x_out, x_trace)
    copyto!(theta_out, theta_trace)
    copyto!(spikes_out, spike_trace)
    return (state.x, state.theta, count)
end

"""PythonCall-stable alias for the in-place batch function."""
simulate_sc_adaptive_threshold_map_b(args...) = simulate_sc_adaptive_threshold_map!(args...)

end # module SCAdaptiveThresholdMapNeuronAccel
