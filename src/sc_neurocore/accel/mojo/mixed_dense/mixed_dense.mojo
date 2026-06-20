# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo batch mixed-precision Q8.8×Q16.16 dense MAC

# Bit-exact Mojo port of the integer mixed-precision dense MAC in
# src/sc_neurocore/compiler/mixed_dense_kernel.py and engine/src/ir/qformat.rs.
#
# Q8.8 weights contract Q16.16 input codes in an Int64 accumulator (the caller
# keeps the contraction within Int64 range); the accumulator divides by the Q8.8
# weight scale with an arithmetic right shift (floor division) and saturates to
# the Q16.16 code range. The arithmetic is exact integer — no transcendental
# path — so this backend is bit-identical to the Rust, Julia, Go and Python
# references.
#
# Mojo @export rejects parametric signatures, so every numpy buffer arrives as a
# raw Int address; reconstruct with `UnsafePointer[T, MutAnyOrigin](
# unsafe_from_address=addr)` inside the function body.

from std.memory import UnsafePointer

comptime WEIGHT_FRACTION = 8
comptime I32_MAX = Int64(2147483647)
comptime I32_MIN = Int64(-2147483648)


@export
fn mixed_dense_forward_batch_q88_q1616_c(
    n_outputs: Int,
    n_inputs: Int,
    n_batch: Int,
    weights_addr: Int,
    inputs_addr: Int,
    outputs_addr: Int,
    overflow_addr: Int,
    underflow_addr: Int,
) -> Int:
    var weights = UnsafePointer[Int16, MutAnyOrigin](unsafe_from_address=weights_addr)
    var inputs = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=inputs_addr)
    var outputs = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=outputs_addr)
    var overflow = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=overflow_addr)
    var underflow = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=underflow_addr
    )

    if n_outputs <= 0 or n_inputs <= 0 or n_batch <= 0:
        return 1

    for b in range(n_batch):
        var input_row = b * n_inputs
        for o in range(n_outputs):
            var weight_row = o * n_inputs
            var sum = Int64(0)
            for i in range(n_inputs):
                sum += Int64(weights[weight_row + i]) * Int64(inputs[input_row + i])
            var scaled = sum >> WEIGHT_FRACTION
            var idx = b * n_outputs + o
            if scaled > I32_MAX:
                outputs[idx] = Int32(I32_MAX)
                overflow[idx] = 1
                underflow[idx] = 0
            elif scaled < I32_MIN:
                outputs[idx] = Int32(I32_MIN)
                overflow[idx] = 1
                underflow[idx] = 0
            else:
                outputs[idx] = Int32(scaled)
                overflow[idx] = 0
                underflow[idx] = UInt8(1) if (sum != 0 and scaled == 0) else UInt8(0)

    return 0
