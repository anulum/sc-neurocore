# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia QCFS quantisation and surrogate derivatives

"""Shifted clipped QCFS quantisation with the Python scalar and array contract."""
module QcfsAccel

using Printf

"""Positive integer rate grid and finite threshold; mutable for training updates."""
mutable struct QCFSActivationState
    T::Int
    theta::Float64

    """Construct a checked grid, defaulting to eight steps and unit threshold."""
    function QCFSActivationState(T::Int=8, theta::Float64=1.0)
        T > 0 || throw(ArgumentError("QCFS T must be a positive integer"))
        isfinite(theta) && theta > 0 ||
            throw(ArgumentError("QCFS theta must be finite and positive"))
        new(T, theta)
    end
end

"""Quantise scalar or array values; saturate infinities and propagate NaNs."""
function forward(s::QCFSActivationState, x)
    s.T > 0 || throw(ArgumentError("QCFS T must be a positive integer"))
    isfinite(s.theta) && s.theta > 0 ||
        throw(ArgumentError("QCFS theta must be finite and positive"))
    shifted = x .* s.T ./ s.theta .+ 0.5
    return floor.(clamp.(shifted, 0.0, Float64(s.T))) .* s.theta ./ s.T
end

"""Describe the active grid and threshold with two decimal places."""
function extra_repr(s::QCFSActivationState)
    return @sprintf("T=%d, theta=%.2f", s.T, s.theta)
end

"""
    backward(s, x, upstream) -> (input_gradient, threshold_gradient)

Straight-through derivatives of one element (Bu et al., 2022, Eq. 17): the
input derivative is `upstream` on the open interior `0 < s < T` and zero
elsewhere; the threshold derivative is the one a one-element batch receives,
evaluated in the Python autograd operation order.
"""
function backward(s::QCFSActivationState, x::Float64, upstream::Float64)
    s.T > 0 || throw(ArgumentError("QCFS T must be a positive integer"))
    isfinite(s.theta) && s.theta > 0 ||
        throw(ArgumentError("QCFS theta must be finite and positive"))
    steps = Float64(s.T)
    theta = s.theta
    shifted = x * steps / theta + 0.5
    interior = shifted > 0 && shifted < steps
    lattice = floor(clamp(shifted, 0.0, steps))
    output_gradient = upstream / steps
    carried = interior ? output_gradient * theta : 0.0
    scaled_input = (interior ? x : 0.0) * steps
    input_gradient = interior ? carried / theta * steps : 0.0
    threshold = (0.0 + output_gradient * lattice) +
                (0.0 + (-carried) * (scaled_input / theta / theta))
    return (input_gradient, threshold)
end

const ADDRESS_LIMIT = UInt(typemax(Int))

"""Admit one complete aligned float64 span in the signed address domain."""
function admitted_span(address::UInt, count::UInt)
    count == 0 && return true
    return address != 0 && address % 8 == 0 && count <= ADDRESS_LIMIT ÷ 8 &&
           address <= ADDRESS_LIMIT - count * 8
end

"""Admit the shared uint32 step domain and a finite positive threshold."""
function admitted_state(steps::Integer, theta::Real)
    1 <= steps <= typemax(UInt32) || return nothing
    threshold = Float64(theta)
    isfinite(threshold) && threshold > 0 || return nothing
    return QCFSActivationState(Int(steps), threshold)
end

"""Report the exact QCFS array ABI version."""
sc_qcfs_abi_version()::UInt32 = 1

"""
    sc_qcfs_forward(steps, theta, x_addr, count, output_addr) -> Int32

Quantise `count` activations between live aligned float64 arrays; returns 0,
or -1 for an invalid grid, threshold or span with every output unchanged.
"""
function sc_qcfs_forward(steps::Integer, theta::Real, x_addr::Integer, count::Integer,
                         output_addr::Integer)::Int32
    state = admitted_state(steps, theta)
    elements = UInt(count)
    state === nothing && return -1
    admitted_span(UInt(x_addr), elements) && admitted_span(UInt(output_addr), elements) ||
        return -1
    x, output = Ptr{Float64}(UInt(x_addr)), Ptr{Float64}(UInt(output_addr))
    for index in 1:Int(elements)
        unsafe_store!(output, forward(state, unsafe_load(x, index)), index)
    end
    return 0
end

"""
    sc_qcfs_backward(steps, theta, x_addr, upstream_addr, count, input_addr, threshold_addr) -> Int32

Write each element's input and threshold derivative; returns 0, or -1 for an
invalid grid, threshold or span with every output unchanged. The two outputs
must not overlap.
"""
function sc_qcfs_backward(steps::Integer, theta::Real, x_addr::Integer, upstream_addr::Integer,
                          count::Integer, input_addr::Integer, threshold_addr::Integer)::Int32
    state = admitted_state(steps, theta)
    elements = UInt(count)
    state === nothing && return -1
    all(address -> admitted_span(UInt(address), elements),
        (x_addr, upstream_addr, input_addr, threshold_addr)) || return -1
    x, upstream = Ptr{Float64}(UInt(x_addr)), Ptr{Float64}(UInt(upstream_addr))
    inputs, thresholds = Ptr{Float64}(UInt(input_addr)), Ptr{Float64}(UInt(threshold_addr))
    for index in 1:Int(elements)
        input, threshold = backward(state, unsafe_load(x, index), unsafe_load(upstream, index))
        unsafe_store!(inputs, input, index)
        unsafe_store!(thresholds, threshold, index)
    end
    return 0
end

end # module QcfsAccel
