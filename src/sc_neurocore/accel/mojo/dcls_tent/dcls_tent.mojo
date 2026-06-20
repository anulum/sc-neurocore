# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo batch DCLS-max Q8.8 tent kernel

# Bit-exact Mojo port of the DCLS-max triangular (tent) weighting kernel in
# src/sc_neurocore/scpn/dcls_tent_kernel.py and engine/src/scpn/dcls.rs
# (Khalfaoui-Hassani, Pellegrini & Masquelier 2023, NeurIPS).
#
# The kernel is exact integer Q8.8 arithmetic — no exp/transcendental path — so
# this backend agrees with the Rust, Julia, Go and Python references bit-for-bit
# (parity tolerance zero), unlike floating-point Mojo kernels that tolerate
# last-ULP libm drift.
#
# Mojo @export rejects parametric signatures, so every numpy buffer arrives as a
# raw Int address; reconstruct with `UnsafePointer[T, MutAnyOrigin](
# unsafe_from_address=addr)` inside the function body.

from std.memory import UnsafePointer

comptime FRACTION = 8
comptime Q88_ONE = Int64(256)
comptime I16_MAX = Int64(32767)
comptime I16_MIN = Int64(-32768)
comptime I32_MAX = Int64(2147483647)
comptime I32_MIN = Int64(-2147483648)
comptime I16_MAX_Q16_16 = Int64(8388352)  # 32767 << 8
comptime I16_MIN_Q16_16 = Int64(-8388608)  # -32768 << 8


@always_inline
fn tent_gate_q88(tap_index: Int64, centre_q88: Int64, sigma_q88: Int64) -> Int64:
    var delay = tap_index << FRACTION
    var distance = delay - centre_q88
    if distance < 0:
        distance = -distance
    if distance >= sigma_q88:
        return Int64(0)
    var gate = ((sigma_q88 - distance) << FRACTION) // sigma_q88
    if gate > Q88_ONE:
        return Q88_ONE
    if gate < 0:
        return Int64(0)
    return gate


@export
fn dcls_max_forward_batch_q88_c(
    n_channels: Int,
    n_taps: Int,
    spikes_addr: Int,
    weights_addr: Int,
    centres_addr: Int,
    sigmas_addr: Int,
    outputs_addr: Int,
    accumulators_addr: Int,
    overflow_addr: Int,
    active_addr: Int,
    max_gates_addr: Int,
) -> Int:
    var spikes = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=spikes_addr)
    var weights = UnsafePointer[Int16, MutAnyOrigin](unsafe_from_address=weights_addr)
    var centres = UnsafePointer[Int16, MutAnyOrigin](unsafe_from_address=centres_addr)
    var sigmas = UnsafePointer[Int16, MutAnyOrigin](unsafe_from_address=sigmas_addr)
    var outputs = UnsafePointer[Int16, MutAnyOrigin](unsafe_from_address=outputs_addr)
    var accumulators = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=accumulators_addr
    )
    var overflow = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=overflow_addr)
    var active = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=active_addr)
    var max_gates = UnsafePointer[Int16, MutAnyOrigin](
        unsafe_from_address=max_gates_addr
    )

    if n_channels <= 0 or n_taps <= 0:
        return 1

    for c in range(n_channels):
        var centre = Int64(centres[c])
        var sigma = Int64(sigmas[c])
        if sigma <= 0:
            return 2
        var base = c * n_taps
        var accumulator = Int64(0)
        var active_count = Int64(0)
        var max_gate = Int64(0)
        for t in range(n_taps):
            if spikes[base + t] == 0:
                continue
            active_count += 1
            var gate = tent_gate_q88(Int64(t), centre, sigma)
            if gate > max_gate:
                max_gate = gate
            accumulator += Int64(weights[base + t]) * gate

        var accumulator_q16_16 = accumulator
        if accumulator_q16_16 > I32_MAX:
            accumulator_q16_16 = I32_MAX
        elif accumulator_q16_16 < I32_MIN:
            accumulator_q16_16 = I32_MIN
        var overflowed = accumulator_q16_16 != accumulator
        var output: Int64
        if accumulator > I16_MAX_Q16_16:
            output = I16_MAX
            overflowed = True
        elif accumulator < I16_MIN_Q16_16:
            output = I16_MIN
            overflowed = True
        else:
            output = accumulator >> FRACTION

        outputs[c] = Int16(output)
        accumulators[c] = Int32(accumulator_q16_16)
        overflow[c] = UInt8(1) if overflowed else UInt8(0)
        active[c] = active_count
        max_gates[c] = Int16(max_gate)

    return 0
