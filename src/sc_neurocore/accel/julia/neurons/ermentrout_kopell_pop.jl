# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia batch mirror for the MPR mean field

"""
Restore `R = tau * r` and `t' = t / tau` from dimensionless equations
(12a-b), then apply simultaneous explicit-Euler updates.
"""
module ErmentroutKopellPopAccel

using PythonCall: PyArray

export is_candidate_error,
    is_configuration_error,
    simulate_ermentrout_kopell_pop!,
    validate_ermentrout_kopell_pop

"""Caller configuration, input, or buffer violation."""
struct MPRConfigurationError <: Exception
    message::String
end

"""Finite-entry recurrence whose simultaneous candidate is invalid."""
struct MPRCandidateError <: Exception
    message::String
end

Base.showerror(io::IO, error::MPRConfigurationError) = print(io, error.message)
Base.showerror(io::IO, error::MPRCandidateError) = print(io, error.message)

is_configuration_error(error)::Bool = error isa MPRConfigurationError
is_candidate_error(error)::Bool = error isa MPRCandidateError

function validate_ermentrout_kopell_pop(values::NTuple{7, Float64})::Bool
    return all(isfinite, values) &&
        values[1] >= 0.0 && values[3] > 0.0 &&
        values[4] >= 0.0 && values[7] > 0.0
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
        throw(MPRConfigurationError("$name must have Float64 elements"))
    length(buffer) == steps ||
        throw(MPRConfigurationError("$name length mismatch"))
    (isempty(buffer) || (applicable(stride, buffer, 1) && stride(buffer, 1) == 1)) ||
        throw(MPRConfigurationError("$name must have unit stride"))
    applicable(pointer, buffer) ||
        throw(MPRConfigurationError("$name must expose contiguous storage"))
    if writable && !_writable(buffer)
        throw(MPRConfigurationError("$name must be writable"))
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

"""Advance a complete external-drive batch into caller-owned r and v traces."""
function simulate_ermentrout_kopell_pop!(
    r_init::Real,
    v_init::Real,
    tau::Real,
    delta::Real,
    eta_bar::Real,
    coupling::Real,
    dt::Real,
    ext_input::AbstractVector,
    r_out::AbstractVector,
    v_out::AbstractVector,
)
    steps = length(ext_input)
    _validate_buffer(ext_input, "ext_input", steps)
    _validate_buffer(r_out, "r_out", steps; writable = true)
    _validate_buffer(v_out, "v_out", steps; writable = true)
    (_overlap(ext_input, r_out) || _overlap(ext_input, v_out) || _overlap(r_out, v_out)) &&
        throw(MPRConfigurationError("MPR input and output buffers must not overlap"))

    values = Float64.((r_init, v_init, tau, delta, eta_bar, coupling, dt))
    configuration = Tuple(values)
    validate_ermentrout_kopell_pop(configuration) ||
        throw(MPRConfigurationError("invalid MPR numerical configuration"))
    all(isfinite, ext_input) ||
        throw(MPRConfigurationError("ext_input must contain only finite values"))

    r_trace = Vector{Float64}(undef, steps)
    v_trace = Vector{Float64}(undef, steps)
    r, v = values[1], values[2]
    tau_f, delta_f = values[3], values[4]
    eta_f, coupling_f, step_size = values[5], values[6], values[7]
    @inbounds for step in 1:steps
        scaled_rate = pi * tau_f * r
        dr = delta_f / (pi * tau_f * tau_f) + 2.0 * r * v / tau_f
        dv = (
            v * v + eta_f + Float64(ext_input[step]) + coupling_f * tau_f * r -
            scaled_rate * scaled_rate
        ) / tau_f
        next_r = r + step_size * dr
        next_v = v + step_size * dv
        isfinite(next_r) && isfinite(next_v) && next_r >= 0.0 ||
            throw(MPRCandidateError("invalid MPR candidate state"))
        r, v = next_r, next_v
        r_trace[step], v_trace[step] = r, v
    end
    copyto!(r_out, r_trace)
    copyto!(v_out, v_trace)
    return (r, v)
end

end # module ErmentroutKopellPopAccel
