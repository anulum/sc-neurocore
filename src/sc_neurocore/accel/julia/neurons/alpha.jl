# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for dual alpha-synapse LIF

module AlphaAccel

using PythonCall: PyArray

export AlphaCandidateError,
    AlphaConfigurationError,
    AlphaNeuronState,
    drive_contribution,
    filter_candidates,
    is_candidate_error,
    is_configuration_error,
    reset!,
    simulate,
    simulate_alpha_b,
    simulate_alpha!,
    step!,
    valid

"""Caller configuration, input, or buffer violation."""
struct AlphaConfigurationError <: Exception
    message::String
end

"""Finite-entry recurrence whose exact-flow candidate is invalid."""
struct AlphaCandidateError <: Exception
    message::String
end

Base.showerror(io::IO, error::AlphaConfigurationError) = print(io, error.message)
Base.showerror(io::IO, error::AlphaCandidateError) = print(io, error.message)

is_configuration_error(error)::Bool = error isa AlphaConfigurationError
is_candidate_error(error)::Bool = error isa AlphaCandidateError

mutable struct AlphaNeuronState
    v::Float64
    a_exc::Float64
    i_exc::Float64
    a_inh::Float64
    i_inh::Float64
    v_rest::Float64
    v_threshold::Float64
    tau_v::Float64
    tau_exc::Float64
    tau_inh::Float64
    dt::Float64
end

AlphaNeuronState() =
    AlphaNeuronState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 5.0, 10.0, 1.0)

function valid(s::AlphaNeuronState)::Bool
    return all(isfinite, (
        s.v,
        s.a_exc,
        s.i_exc,
        s.a_inh,
        s.i_inh,
        s.v_rest,
        s.v_threshold,
        s.tau_v,
        s.tau_exc,
        s.tau_inh,
        s.dt,
    )) &&
        s.tau_v > 0.0 &&
        s.tau_exc > 0.0 &&
        s.tau_inh > 0.0 &&
        s.dt > 0.0 &&
        s.v_threshold > s.v_rest
end

function filter_candidates(
    rise_state::Float64,
    current_state::Float64,
    drive::Float64,
    tau::Float64,
    dt::Float64,
)
    steady_state = tau * drive
    rise_delta = rise_state - steady_state
    current_delta = current_state - steady_state
    decay = exp(-dt / tau)
    rise_next = steady_state + rise_delta * decay
    current_next = steady_state + decay * (current_delta + rise_delta * dt / tau)
    (isfinite(rise_next) && isfinite(current_next)) ||
        throw(AlphaCandidateError("alpha exact-flow filter candidate must remain finite"))
    return rise_next, current_next
end

function drive_contribution(
    current_delta::Float64,
    rise_delta::Float64,
    tau_drive::Float64,
    tau_v::Float64,
    dt::Float64,
)
    rate_v = 1.0 / tau_v
    rate_drive = 1.0 / tau_drive
    decay_v = exp(-dt / tau_v)
    decay_drive = exp(-dt / tau_drive)
    contribution = if abs(rate_v - rate_drive) <= 1.0e-14
        rate_v * decay_v * (current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive))
    else
        rate_delta = rate_v - rate_drive
        first_order = current_delta * (decay_drive - decay_v) / rate_delta
        second_order = rise_delta / tau_drive *
            (decay_drive * (rate_delta * dt - 1.0) + decay_v) /
            (rate_delta * rate_delta)
        rate_v * (first_order + second_order)
    end
    isfinite(contribution) ||
        throw(AlphaCandidateError("alpha exact-flow convolution must remain finite"))
    return contribution
end

function step!(s::AlphaNeuronState, exc_current::Real = 0.0, inh_current::Real = 0.0)::Int
    exc_drive = Float64(exc_current)
    inh_drive = Float64(inh_current)
    (isfinite(exc_drive) && isfinite(inh_drive) && valid(s)) ||
        throw(AlphaConfigurationError(
            "alpha state/current must be finite and well-formed",
        ))
    a_exc_next, i_exc_next = filter_candidates(s.a_exc, s.i_exc, exc_drive, s.tau_exc, s.dt)
    a_inh_next, i_inh_next = filter_candidates(s.a_inh, s.i_inh, inh_drive, s.tau_inh, s.dt)
    exc_steady = s.tau_exc * exc_drive
    inh_steady = s.tau_inh * inh_drive
    v_steady = s.v_rest + exc_steady - inh_steady
    decay_v = exp(-s.dt / s.tau_v)
    v_next = v_steady +
        (s.v - v_steady) * decay_v +
        drive_contribution(s.i_exc - exc_steady, s.a_exc - exc_steady, s.tau_exc, s.tau_v, s.dt) -
        drive_contribution(s.i_inh - inh_steady, s.a_inh - inh_steady, s.tau_inh, s.tau_v, s.dt)
    isfinite(v_next) ||
        throw(AlphaCandidateError("alpha exact-flow candidate must remain finite"))
    s.a_exc, s.i_exc = a_exc_next, i_exc_next
    s.a_inh, s.i_inh = a_inh_next, i_inh_next
    if v_next >= s.v_threshold
        s.v = s.v_rest
        return 1
    end
    s.v = v_next
    return 0
end

function reset!(s::AlphaNeuronState)::Nothing
    s.v = s.v_rest
    s.a_exc = 0.0
    s.i_exc = 0.0
    s.a_inh = 0.0
    s.i_inh = 0.0
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
        throw(AlphaConfigurationError("$name must have Float64 elements"))
    length(buffer) == steps ||
        throw(AlphaConfigurationError("$name length mismatch"))
    (isempty(buffer) || (applicable(stride, buffer, 1) && stride(buffer, 1) == 1)) ||
        throw(AlphaConfigurationError("$name must have unit stride"))
    applicable(pointer, buffer) ||
        throw(AlphaConfigurationError("$name must expose contiguous storage"))
    if writable && !_writable(buffer)
        throw(AlphaConfigurationError("$name must be writable"))
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

function _buffers_distinct(buffers::Tuple{Vararg{AbstractVector}})::Bool
    for left in 1:(length(buffers) - 1)
        for right in (left + 1):length(buffers)
            if _overlap(buffers[left], buffers[right])
                return false
            end
        end
    end
    return true
end

"""Advance a complete piecewise-constant drive batch into caller-owned buffers."""
function simulate_alpha!(
    v_init::Real,
    a_exc_init::Real,
    i_exc_init::Real,
    a_inh_init::Real,
    i_inh_init::Real,
    v_rest::Real,
    v_threshold::Real,
    tau_v::Real,
    tau_exc::Real,
    tau_inh::Real,
    dt::Real,
    exc_current::AbstractVector,
    inh_current::AbstractVector,
    v_out::AbstractVector,
    a_exc_out::AbstractVector,
    i_exc_out::AbstractVector,
    a_inh_out::AbstractVector,
    i_inh_out::AbstractVector,
    spikes_out::AbstractVector,
)
    steps = length(exc_current)
    length(inh_current) == steps ||
        throw(AlphaConfigurationError("inh_current length mismatch"))
    _validate_buffer(exc_current, "exc_current", steps)
    _validate_buffer(inh_current, "inh_current", steps)
    _validate_buffer(v_out, "v_out", steps; writable = true)
    _validate_buffer(a_exc_out, "a_exc_out", steps; writable = true)
    _validate_buffer(i_exc_out, "i_exc_out", steps; writable = true)
    _validate_buffer(a_inh_out, "a_inh_out", steps; writable = true)
    _validate_buffer(i_inh_out, "i_inh_out", steps; writable = true)
    _validate_buffer(spikes_out, "spikes_out", steps; writable = true)
    _buffers_distinct((
        exc_current,
        inh_current,
        v_out,
        a_exc_out,
        i_exc_out,
        a_inh_out,
        i_inh_out,
        spikes_out,
    )) ||
        throw(AlphaConfigurationError(
            "alpha input and output buffers must not overlap",
        ))

    values = Float64.((v_init, a_exc_init, i_exc_init, a_inh_init, i_inh_init, v_rest, v_threshold, tau_v, tau_exc, tau_inh, dt))
    state = AlphaNeuronState(values...)
    valid(state) ||
        throw(AlphaConfigurationError("invalid alpha numerical configuration"))
    (all(isfinite, exc_current) && all(isfinite, inh_current)) ||
        throw(AlphaConfigurationError("current values must contain only finite values"))

    traces = (
        Vector{Float64}(undef, steps),
        Vector{Float64}(undef, steps),
        Vector{Float64}(undef, steps),
        Vector{Float64}(undef, steps),
        Vector{Float64}(undef, steps),
        Vector{Float64}(undef, steps),
    )
    spike_count = 0
    @inbounds for index in 1:steps
        spike = step!(state, Float64(exc_current[index]), Float64(inh_current[index]))
        traces[1][index] = state.v
        traces[2][index] = state.a_exc
        traces[3][index] = state.i_exc
        traces[4][index] = state.a_inh
        traces[5][index] = state.i_inh
        traces[6][index] = Float64(spike)
        spike_count += spike
    end
    copyto!(v_out, traces[1])
    copyto!(a_exc_out, traces[2])
    copyto!(i_exc_out, traces[3])
    copyto!(a_inh_out, traces[4])
    copyto!(i_inh_out, traces[5])
    copyto!(spikes_out, traces[6])
    return (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh, spike_count)
end

"""PythonCall-safe alias for the mutating complete-batch entry point."""
simulate_alpha_b(args...) = simulate_alpha!(args...)

"""Compatibility helper for a constant drive and catalogue-default configuration."""
function simulate(n_steps::Int = 1000; I_ext::Float64 = 1.0, dt::Float64 = 1.0)
    n_steps >= 0 ||
        throw(AlphaConfigurationError("n_steps must be non-negative"))
    exc_current = fill(I_ext, n_steps)
    inh_current = zeros(n_steps)
    v_trace = zeros(n_steps)
    a_exc_trace = zeros(n_steps)
    i_exc_trace = zeros(n_steps)
    a_inh_trace = zeros(n_steps)
    i_inh_trace = zeros(n_steps)
    spike_trace = zeros(n_steps)
    _, _, _, _, _, spike_count = simulate_alpha!(
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 20.0, 5.0, 10.0, dt,
        exc_current, inh_current,
        v_trace, a_exc_trace, i_exc_trace, a_inh_trace, i_inh_trace, spike_trace,
    )
    return v_trace, spike_count
end

end # module AlphaAccel
