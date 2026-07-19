# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for composite reduced adaptive-threshold IF

module AdaptiveThresholdIFAccel

using PythonCall: PyArray

export AdaptiveThresholdIFCandidateError,
    AdaptiveThresholdIFConfigurationError,
    AdaptiveThresholdIFNeuronState,
    exact_relaxation,
    is_candidate_error,
    is_configuration_error,
    reset!,
    simulate,
    simulate_adaptive_threshold_if_b,
    simulate_adaptive_threshold_if!,
    step!,
    valid

"""Caller configuration, input, or buffer violation."""
struct AdaptiveThresholdIFConfigurationError <: Exception
    message::String
end

"""Finite-entry recurrence whose exact-relaxation candidate is invalid."""
struct AdaptiveThresholdIFCandidateError <: Exception
    message::String
end

Base.showerror(io::IO, error::AdaptiveThresholdIFConfigurationError) = print(io, error.message)
Base.showerror(io::IO, error::AdaptiveThresholdIFCandidateError) = print(io, error.message)

is_configuration_error(error)::Bool = error isa AdaptiveThresholdIFConfigurationError
is_candidate_error(error)::Bool = error isa AdaptiveThresholdIFCandidateError

mutable struct AdaptiveThresholdIFNeuronState
    v::Float64
    theta::Float64
    v_rest::Float64
    v_reset::Float64
    theta_rest::Float64
    delta_theta::Float64
    tau_m::Float64
    tau_theta::Float64
    dt::Float64
end

AdaptiveThresholdIFNeuronState() =
    AdaptiveThresholdIFNeuronState(-65.0, -50.0, -65.0, -65.0, -50.0, 5.0, 10.0, 50.0, 0.1)

function valid(s::AdaptiveThresholdIFNeuronState)::Bool
    return all(isfinite, (
        s.v,
        s.theta,
        s.v_rest,
        s.v_reset,
        s.theta_rest,
        s.delta_theta,
        s.tau_m,
        s.tau_theta,
        s.dt,
    )) &&
        s.delta_theta >= 0.0 &&
        s.tau_m > 0.0 &&
        s.tau_theta > 0.0 &&
        s.dt > 0.0 &&
        s.theta_rest > s.v_rest &&
        s.theta_rest > s.v_reset
end

function exact_relaxation(state::Float64, steady_state::Float64, tau::Float64, dt::Float64)
    decay = exp(-dt / tau)
    candidate = steady_state + (state - steady_state) * decay
    isfinite(candidate) ||
        throw(AdaptiveThresholdIFCandidateError(
            "adaptive-threshold exact-relaxation candidate must remain finite",
        ))
    return candidate
end

function step!(s::AdaptiveThresholdIFNeuronState, current::Real = 0.0)::Int
    drive = Float64(current)
    isfinite(drive) && valid(s) ||
        throw(AdaptiveThresholdIFConfigurationError(
            "adaptive-threshold state/current must be finite and well-formed",
        ))
    next_v = exact_relaxation(s.v, s.v_rest + drive, s.tau_m, s.dt)
    next_theta = exact_relaxation(s.theta, s.theta_rest, s.tau_theta, s.dt)
    if next_v >= next_theta
        spike_theta = next_theta + s.delta_theta
        isfinite(spike_theta) ||
            throw(AdaptiveThresholdIFCandidateError(
                "adaptive-threshold threshold jump must remain finite",
            ))
        s.v = s.v_reset
        s.theta = spike_theta
        return 1
    end
    s.v, s.theta = next_v, next_theta
    return 0
end

function reset!(s::AdaptiveThresholdIFNeuronState)::Nothing
    s.v = s.v_rest
    s.theta = s.theta_rest
    return nothing
end

_writable(buffer::AbstractVector{Float64}) =
    applicable(setindex!, buffer, 0.0, firstindex(buffer))
_writable(::PyArray{Float64, 1, W, C, R}) where {W, C, R} = W

function _validate_buffer(
    buffer::AbstractVector,
    name::String,
    steps::Int;
    writable::Bool = false,
)
    eltype(buffer) === Float64 ||
        throw(AdaptiveThresholdIFConfigurationError("$name must have Float64 elements"))
    length(buffer) == steps ||
        throw(AdaptiveThresholdIFConfigurationError("$name length mismatch"))
    (isempty(buffer) || (applicable(stride, buffer, 1) && stride(buffer, 1) == 1)) ||
        throw(AdaptiveThresholdIFConfigurationError("$name must have unit stride"))
    applicable(pointer, buffer) ||
        throw(AdaptiveThresholdIFConfigurationError("$name must expose contiguous storage"))
    if writable && !_writable(buffer)
        throw(AdaptiveThresholdIFConfigurationError("$name must be writable"))
    end
    return nothing
end

function _overlap(a::AbstractVector, b::AbstractVector)::Bool
    if isempty(a) || isempty(b)
        return false
    end
    a_start = UInt(pointer(a))
    b_start = UInt(pointer(b))
    a_bytes = UInt(length(a)) * UInt(sizeof(eltype(a)))
    b_bytes = UInt(length(b)) * UInt(sizeof(eltype(b)))
    return a_start <= b_start ?
        b_start - a_start < a_bytes :
        a_start - b_start < b_bytes
end

function _buffers_distinct(buffers::NTuple{4, AbstractVector})::Bool
    for left in 1:(length(buffers) - 1)
        for right in (left + 1):length(buffers)
            if _overlap(buffers[left], buffers[right])
                return false
            end
        end
    end
    return true
end

"""Advance a complete piecewise-constant current vector into caller-owned buffers."""
function simulate_adaptive_threshold_if!(
    v_init::Real,
    theta_init::Real,
    v_rest::Real,
    v_reset::Real,
    theta_rest::Real,
    delta_theta::Real,
    tau_m::Real,
    tau_theta::Real,
    dt::Real,
    current::AbstractVector,
    v_out::AbstractVector,
    theta_out::AbstractVector,
    spikes_out::AbstractVector,
)
    steps = length(current)
    _validate_buffer(current, "current", steps)
    _validate_buffer(v_out, "v_out", steps; writable = true)
    _validate_buffer(theta_out, "theta_out", steps; writable = true)
    _validate_buffer(spikes_out, "spikes_out", steps; writable = true)
    _buffers_distinct((current, v_out, theta_out, spikes_out)) ||
        throw(AdaptiveThresholdIFConfigurationError(
            "adaptive-threshold input and output buffers must not overlap",
        ))

    values = Float64.((v_init, theta_init, v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt))
    state = AdaptiveThresholdIFNeuronState(values...)
    valid(state) ||
        throw(AdaptiveThresholdIFConfigurationError(
            "invalid adaptive-threshold numerical configuration",
        ))
    all(isfinite, current) ||
        throw(AdaptiveThresholdIFConfigurationError(
            "current must contain only finite values",
        ))

    v_trace = Vector{Float64}(undef, steps)
    theta_trace = Vector{Float64}(undef, steps)
    spike_trace = Vector{Float64}(undef, steps)
    spike_count = 0
    @inbounds for index in 1:steps
        spike = step!(state, Float64(current[index]))
        v_trace[index] = state.v
        theta_trace[index] = state.theta
        spike_trace[index] = Float64(spike)
        spike_count += spike
    end
    copyto!(v_out, v_trace)
    copyto!(theta_out, theta_trace)
    copyto!(spikes_out, spike_trace)
    return (state.v, state.theta, spike_count)
end

"""PythonCall-safe alias for the mutating complete-batch entry point."""
simulate_adaptive_threshold_if_b(args...) = simulate_adaptive_threshold_if!(args...)

"""Compatibility helper for a constant current and catalogue-default configuration."""
function simulate(n_steps::Int = 1000; I_ext::Float64 = 0.0, dt::Float64 = 0.1)
    n_steps >= 0 ||
        throw(AdaptiveThresholdIFConfigurationError("n_steps must be non-negative"))
    current = fill(I_ext, n_steps)
    v_trace = zeros(n_steps)
    theta_trace = zeros(n_steps)
    spike_trace = zeros(n_steps)
    _, _, spike_count = simulate_adaptive_threshold_if!(
        -65.0,
        -50.0,
        -65.0,
        -65.0,
        -50.0,
        5.0,
        10.0,
        50.0,
        dt,
        current,
        v_trace,
        theta_trace,
        spike_trace,
    )
    return v_trace, spike_count
end

end # module AdaptiveThresholdIFAccel
