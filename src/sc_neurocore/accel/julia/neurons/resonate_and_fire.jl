# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for Izhikevich resonate-and-fire

module ResonateAndFireAccel

using PythonCall: PyArray

export ResonateAndFireCandidateError,
    ResonateAndFireConfigurationError,
    ResonateAndFireNeuronState,
    exact_flow,
    is_candidate_error,
    is_configuration_error,
    reset!,
    simulate,
    simulate_resonate_and_fire_b,
    simulate_resonate_and_fire!,
    step!,
    valid

"""Caller configuration, input, or buffer violation."""
struct ResonateAndFireConfigurationError <: Exception
    message::String
end

"""Finite-entry recurrence whose exact-flow candidate is invalid."""
struct ResonateAndFireCandidateError <: Exception
    message::String
end

Base.showerror(io::IO, error::ResonateAndFireConfigurationError) = print(io, error.message)
Base.showerror(io::IO, error::ResonateAndFireCandidateError) = print(io, error.message)

is_configuration_error(error)::Bool = error isa ResonateAndFireConfigurationError
is_candidate_error(error)::Bool = error isa ResonateAndFireCandidateError

mutable struct ResonateAndFireNeuronState
    x::Float64
    y::Float64
    b::Float64
    omega::Float64
    threshold::Float64
    dt::Float64
end

ResonateAndFireNeuronState() =
    ResonateAndFireNeuronState(0.0, 0.0, -1.0, 10.0, 1.0, 0.01)

function valid(s::ResonateAndFireNeuronState)::Bool
    return all(isfinite, (s.x, s.y, s.b, s.omega, s.threshold, s.dt)) &&
        s.omega > 0.0 && s.threshold > 0.0 && s.dt > 0.0
end

function exact_flow(
    x::Float64,
    y::Float64,
    current::Float64,
    b::Float64,
    omega::Float64,
    dt::Float64,
)
    denominator = b * b + omega * omega
    damping_argument = b * dt
    angle = omega * dt
    x_ss = -b * current / denominator
    y_ss = omega * current / denominator
    decay = exp(damping_argument)
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    all(isfinite, (
        denominator,
        damping_argument,
        angle,
        x_ss,
        y_ss,
        decay,
        cos_angle,
        sin_angle,
    )) && denominator > 0.0 ||
        throw(ResonateAndFireCandidateError(
            "resonate-and-fire exact-flow coefficients must remain finite",
        ))
    dx = x - x_ss
    dy = y - y_ss
    next_x = x_ss + decay * (dx * cos_angle - dy * sin_angle)
    next_y = y_ss + decay * (dx * sin_angle + dy * cos_angle)
    isfinite(next_x) && isfinite(next_y) ||
        throw(ResonateAndFireCandidateError(
            "resonate-and-fire exact-flow candidate must remain finite",
        ))
    return next_x, next_y
end

function step!(s::ResonateAndFireNeuronState, current::Real = 0.0)::Int
    drive = Float64(current)
    isfinite(drive) && valid(s) ||
        throw(ResonateAndFireConfigurationError(
            "resonate-and-fire state/current must be finite and well-formed",
        ))
    old_y = s.y
    next_x, next_y = exact_flow(s.x, s.y, drive, s.b, s.omega, s.dt)
    if old_y < s.threshold && next_y >= s.threshold
        s.x = 0.0
        s.y = s.threshold
        return 1
    end
    s.x, s.y = next_x, next_y
    return 0
end

function reset!(s::ResonateAndFireNeuronState)::Nothing
    s.x = 0.0
    s.y = 0.0
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
        throw(ResonateAndFireConfigurationError("$name must have Float64 elements"))
    length(buffer) == steps ||
        throw(ResonateAndFireConfigurationError("$name length mismatch"))
    (isempty(buffer) || (applicable(stride, buffer, 1) && stride(buffer, 1) == 1)) ||
        throw(ResonateAndFireConfigurationError("$name must have unit stride"))
    applicable(pointer, buffer) ||
        throw(ResonateAndFireConfigurationError("$name must expose contiguous storage"))
    if writable && !_writable(buffer)
        throw(ResonateAndFireConfigurationError("$name must be writable"))
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
function simulate_resonate_and_fire!(
    x_init::Real,
    y_init::Real,
    b::Real,
    omega::Real,
    threshold::Real,
    dt::Real,
    current::AbstractVector,
    x_out::AbstractVector,
    y_out::AbstractVector,
    spikes_out::AbstractVector,
)
    steps = length(current)
    _validate_buffer(current, "current", steps)
    _validate_buffer(x_out, "x_out", steps; writable = true)
    _validate_buffer(y_out, "y_out", steps; writable = true)
    _validate_buffer(spikes_out, "spikes_out", steps; writable = true)
    _buffers_distinct((current, x_out, y_out, spikes_out)) ||
        throw(ResonateAndFireConfigurationError(
            "resonate-and-fire input and output buffers must not overlap",
        ))

    values = Float64.((x_init, y_init, b, omega, threshold, dt))
    state = ResonateAndFireNeuronState(values...)
    valid(state) ||
        throw(ResonateAndFireConfigurationError(
            "invalid resonate-and-fire numerical configuration",
        ))
    all(isfinite, current) ||
        throw(ResonateAndFireConfigurationError(
            "current must contain only finite values",
        ))

    x_trace = Vector{Float64}(undef, steps)
    y_trace = Vector{Float64}(undef, steps)
    spike_trace = Vector{Float64}(undef, steps)
    spike_count = 0
    @inbounds for index in 1:steps
        spike = step!(state, Float64(current[index]))
        x_trace[index] = state.x
        y_trace[index] = state.y
        spike_trace[index] = Float64(spike)
        spike_count += spike
    end
    copyto!(x_out, x_trace)
    copyto!(y_out, y_trace)
    copyto!(spikes_out, spike_trace)
    return (state.x, state.y, spike_count)
end

"""PythonCall-safe alias for the mutating complete-batch entry point."""
simulate_resonate_and_fire_b(args...) = simulate_resonate_and_fire!(args...)

"""Compatibility helper for a constant current and source-default configuration."""
function simulate(n_steps::Int = 1000; I_ext::Float64 = 2.0, dt::Float64 = 0.01)
    n_steps >= 0 ||
        throw(ResonateAndFireConfigurationError("n_steps must be non-negative"))
    current = fill(I_ext, n_steps)
    x_trace = zeros(n_steps)
    y_trace = zeros(n_steps)
    spike_trace = zeros(n_steps)
    _, _, spike_count = simulate_resonate_and_fire!(
        0.0,
        0.0,
        -1.0,
        10.0,
        1.0,
        dt,
        current,
        x_trace,
        y_trace,
        spike_trace,
    )
    return y_trace, spike_count
end

end # module ResonateAndFireAccel
