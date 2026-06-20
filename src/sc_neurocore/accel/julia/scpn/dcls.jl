# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia DCLS-max Q8.8 tent kernel (batch)

"""
Bit-exact Julia port of the DCLS-max triangular (tent) weighting kernel in
`src/sc_neurocore/scpn/dcls_tent_kernel.py` and `engine/src/scpn/dcls.rs`
(Khalfaoui-Hassani, Pellegrini & Masquelier 2023, NeurIPS).

The kernel is exact integer Q8.8 arithmetic, so this backend agrees with the
Rust, Go, Mojo and Python references bit-for-bit. The tent gate uses truncating
integer division (`div`) and the output shift is an arithmetic right shift,
matching the synthesisable RTL.
"""
module DclsTentAccel

export dcls_max_forward_batch_q88!, tent_gate_q88

const FRACTION = 8
const Q88_ONE = Int64(1) << FRACTION
const I16_MAX = Int64(32767)
const I16_MIN = Int64(-32768)
const I32_MAX = Int64(2147483647)
const I32_MIN = Int64(-2147483648)
const I16_MAX_Q16_16 = I16_MAX << FRACTION
const I16_MIN_Q16_16 = I16_MIN << FRACTION

"""Q8.8 triangular tent gate for a zero-based delay tap."""
@inline function tent_gate_q88(tap_index::Int64, centre_q88::Int64, sigma_q88::Int64)::Int64
    sigma_q88 > 0 || throw(ArgumentError("DCLS tent sigma must be positive"))
    delay_q88 = tap_index << FRACTION
    distance_q88 = abs(delay_q88 - centre_q88)
    if distance_q88 >= sigma_q88
        return Int64(0)
    end
    gate = div((sigma_q88 - distance_q88) << FRACTION, sigma_q88)
    return min(Q88_ONE, max(Int64(0), gate))
end

"""Saturate a raw accumulator into the (output_q88, accumulator_q16_16, overflow) contract."""
@inline function saturate_contraction(accumulator::Int64)
    accumulator_q16_16 = min(I32_MAX, max(I32_MIN, accumulator))
    accumulator_overflow = accumulator_q16_16 != accumulator
    if accumulator > I16_MAX_Q16_16
        return (I16_MAX, accumulator_q16_16, true)
    elseif accumulator < I16_MIN_Q16_16
        return (I16_MIN, accumulator_q16_16, true)
    end
    return (accumulator >> FRACTION, accumulator_q16_16, accumulator_overflow)
end

"""
Batched DCLS-max contraction; fills the pre-allocated output buffers in place.

Channel `c` (zero-based) occupies the row-major slice `[c·n_taps, (c+1)·n_taps)`
of `spikes`/`weights` and uses learnable `centres[c+1]`/`sigmas[c+1]`. The output
vectors are indexed 1..n_channels and receive the saturated output, accumulator,
saturation flag, active-tap count and largest applied gate.
"""
function dcls_max_forward_batch_q88!(
    spikes::AbstractVector{<:Integer},
    weights::AbstractVector{<:Integer},
    centres::AbstractVector{<:Integer},
    sigmas::AbstractVector{<:Integer},
    n_taps::Integer,
    outputs::AbstractVector{<:Integer},
    accumulators::AbstractVector{<:Integer},
    overflow::AbstractVector{<:Integer},
    active_counts::AbstractVector{<:Integer},
    max_gates::AbstractVector{<:Integer},
)
    taps = Int64(n_taps)
    taps > 0 || throw(ArgumentError("n_taps must be positive"))
    n_channels = length(centres)
    n_channels > 0 || throw(ArgumentError("DCLS batch requires at least one channel"))
    length(sigmas) == n_channels || throw(ArgumentError("centre/sigma length mismatch"))
    expected = n_channels * taps
    (length(spikes) == expected && length(weights) == expected) ||
        throw(ArgumentError("spike/weight length must be n_channels * n_taps"))

    @inbounds for channel in 0:(n_channels - 1)
        centre = Int64(centres[channel + 1])
        sigma = Int64(sigmas[channel + 1])
        sigma > 0 || throw(ArgumentError("every DCLS sigma must be positive"))
        base = channel * taps
        accumulator = Int64(0)
        active = Int64(0)
        max_gate = Int64(0)
        for tap in 0:(taps - 1)
            if spikes[base + tap + 1] == 0
                continue
            end
            active += 1
            gate = tent_gate_q88(tap, centre, sigma)
            if gate > max_gate
                max_gate = gate
            end
            accumulator += Int64(weights[base + tap + 1]) * gate
        end
        output_q88, accumulator_q16_16, overflowed = saturate_contraction(accumulator)
        outputs[channel + 1] = output_q88
        accumulators[channel + 1] = accumulator_q16_16
        overflow[channel + 1] = overflowed ? 1 : 0
        active_counts[channel + 1] = active
        max_gates[channel + 1] = max_gate
    end
    return nothing
end

end # module DclsTentAccel
