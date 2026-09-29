# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Published N-MNIST records and native C interface

"""Decode Orchard et al. (2015) 40-bit records with float64 millisecond times."""

from std.memory import Pointer


@export
def nmnist_decode_c(
    raw_addr: Int, byte_count: Int, output_addr: Int, value_count: Int
) abi("C") -> Int32:
    """Write x, y, polarity and millisecond rows into caller-owned doubles.

    Return zero on success and -1 on invalid sizes, pointers, alignment or
    overlap, before any output mutation. Empty input accepts null addresses.
    Nonempty addresses must identify live allocations of the declared sizes;
    the caller grants immutable input and exclusive writable destination access.
    The Int address/count ABI targets the platform's pointer-sized C arguments.
    """
    if (
        byte_count < 0
        or value_count < 0
        or value_count > 0x7FFFFFFFFFFFFFFF // 8
        or byte_count % 5 != 0
        or value_count % 4 != 0
        or value_count // 4 != byte_count // 5
    ):
        return -1
    if byte_count == 0:
        return 0
    if raw_addr <= 0 or output_addr <= 0 or output_addr % 8 != 0:
        return -1
    if (
        raw_addr > 0x7FFFFFFFFFFFFFFF - byte_count
        or output_addr > 0x7FFFFFFFFFFFFFFF - value_count * 8
    ):
        return -1
    var input_end = raw_addr + byte_count
    var output_end = output_addr + value_count * 8
    if raw_addr < output_end and output_addr < input_end:
        return -1
    var raw = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=raw_addr)
    var output = Pointer[Float64, MutAnyOrigin](unsafe_from_address=output_addr)
    for index in range(byte_count // 5):
        var source = index * 5
        var destination = index * 4
        output[unsafe_offset=destination] = Float64(raw[unsafe_offset=source])
        output[unsafe_offset=destination + 1] = Float64(raw[unsafe_offset=source + 1])
        output[unsafe_offset=destination + 2] = Float64(raw[unsafe_offset=source + 2] >> 7)
        var time_us = (
            (UInt32(raw[unsafe_offset=source + 2] & 0x7F) << 16)
            | (UInt32(raw[unsafe_offset=source + 3]) << 8)
            | UInt32(raw[unsafe_offset=source + 4])
        )
        output[unsafe_offset=destination + 3] = Float64(time_us) / Float64(1000)
    return 0
