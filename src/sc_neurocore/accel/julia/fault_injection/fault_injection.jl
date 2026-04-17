# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia fault injection (parity with FaultInjector.inject)

"""
Julia implementations of the 5 fault injection kernels used by
`sc_neurocore.fault_injection.FaultInjector.inject`. Each kernel
operates on a `Vector{UInt8}` where each element is 0 or 1 (mirroring
a numpy bool array cast to uint8) and returns the corrupted bitstream
plus a count of bits actually changed.

The RNG is `Random.Xoshiro` seeded per call so back-to-back
invocations with the same seed are reproducible. Bitwise parity with
numpy's PCG64 is impossible; parity tests are statistical (counts
within 4σ of Binomial(n, ber) mean — see the bench harness).

Usage (from Python via juliacall):

    from juliacall import Main as jl
    jl.include("src/sc_neurocore/accel/julia/fault_injection/fault_injection.jl")
    out, n = jl.FaultInjectionAccel.inject_bitflip(bitstream, ber, seed)
"""

module FaultInjectionAccel

using Random

"""Bit-flip: each bit flips with probability `ber`.
Returns (corrupted::Vector{UInt8}, n_flipped::UInt64)."""
function inject_bitflip(bitstream::AbstractVector{UInt8}, ber::Real, seed::Integer)
    out = copy(Vector{UInt8}(bitstream))
    ber <= 0.0 && return (out, UInt64(0))
    rng = Xoshiro(UInt64(seed))
    n = UInt64(0)
    @inbounds for i in eachindex(out)
        if rand(rng) < ber
            out[i] ⊻= 0x01
            n += 1
        end
    end
    return (out, n)
end

"""Stuck-at-0: each bit forced to 0 with probability `ber`.
n_changed counts only bits that were originally 1."""
function inject_stuck_at_0(bitstream::AbstractVector{UInt8}, ber::Real, seed::Integer)
    out = copy(Vector{UInt8}(bitstream))
    ber <= 0.0 && return (out, UInt64(0))
    rng = Xoshiro(UInt64(seed))
    affected = UInt64(0)
    @inbounds for i in eachindex(out)
        if rand(rng) < ber
            if out[i] != 0x00
                affected += 1
            end
            out[i] = 0x00
        end
    end
    return (out, affected)
end

"""Stuck-at-1: each bit forced to 1 with probability `ber`.
n_changed counts only bits that were originally 0."""
function inject_stuck_at_1(bitstream::AbstractVector{UInt8}, ber::Real, seed::Integer)
    out = copy(Vector{UInt8}(bitstream))
    ber <= 0.0 && return (out, UInt64(0))
    rng = Xoshiro(UInt64(seed))
    affected = UInt64(0)
    @inbounds for i in eachindex(out)
        if rand(rng) < ber
            if out[i] == 0x00
                affected += 1
            end
            out[i] = 0x01
        end
    end
    return (out, affected)
end

"""Dropout: alias for stuck-at-0 in this fault model."""
function inject_dropout(bitstream::AbstractVector{UInt8}, ber::Real, seed::Integer)
    return inject_stuck_at_0(bitstream, ber, seed)
end

"""Gaussian noise: add N(0, σ=ber) to bitstream cast to Float64,
clip to [0,1], threshold at 0.5. Returns (corrupted, n_flipped)."""
function inject_gaussian(bitstream::AbstractVector{UInt8}, ber::Real, seed::Integer)
    out = copy(Vector{UInt8}(bitstream))
    ber <= 0.0 && return (out, UInt64(0))
    rng = Xoshiro(UInt64(seed))
    flipped = UInt64(0)
    @inbounds for i in eachindex(out)
        original = out[i]
        noisy = clamp(Float64(original) + ber * randn(rng), 0.0, 1.0)
        new_bit::UInt8 = noisy > 0.5 ? 0x01 : 0x00
        if new_bit != original
            flipped += 1
        end
        out[i] = new_bit
    end
    return (out, flipped)
end

end  # module FaultInjectionAccel
