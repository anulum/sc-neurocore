# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native DVS numeric scalar conversion

"""Widen stored real NPY scalars to doubles without time or coordinate coercion."""

from std.collections import List
from std.ffi import external_call
from std.memory import Pointer, bitcast


struct Scalar(Copyable, Movable):
    """Describe one admitted Boolean, integer or floating storage scalar.

    Extended precision uses the host C long-double conversion shared with
    Rust. Its ABI width is checked before decoding. The byte order refers
    to the stored bytes; returned values are native float64.
    """

    var kind: UInt8
    """ASCII storage kind: b (Boolean), i (signed), u (unsigned), f (float)."""
    var width: Int
    """Number of bytes occupied by a stored scalar."""
    var big: Bool
    """Whether the stored scalar uses big-endian byte order."""

    def __init__(out self, kind: UInt8, width: Int, big: Bool) raises:
        """Refuse nonreal kinds and unsupported widths before any scalar read.

        Args:
            kind: ASCII b, i, u or f storage category.
            width: Storage bytes per scalar.
            big: Stored big-endian byte order.

        Raises:
            Error: Invalid kind, width or extended ABI.
        """
        var valid = False
        if kind == UInt8(98):
            valid = width == 1
        elif kind == UInt8(105) or kind == UInt8(117):
            valid = width == 1 or width == 2 or width == 4 or width == 8
        elif kind == UInt8(102):
            valid = width == 2 or width == 4 or width == 8 or width == 16
        if not valid:
            raise Error("nonreal or incompatible DVS scalar")
        if width == 16 and external_call["sc_dvs_extended_width", Int]() != 16:
            raise Error("DVS extended ABI is unavailable")
        self.kind = kind
        self.width = width
        self.big = big

    def decode(self, raw: List[UInt8], offset: Int) raises -> Float64:
        """Read one complete scalar at a checked offset and widen exactly once.

        Preserve signed zero, infinities, subnormals and float64 payload bits.
        Integer conversion follows native IEEE rounding. Boolean storage
        treats any nonzero byte as true. Truncated or negative offsets refuse
        before accessing memory. Geometry and finite-value checks belong to
        the dataset and encoder contracts.

        Args:
            raw: Complete owned scalar input bytes.
            offset: Zero-based scalar start within raw.

        Returns:
            Native double representing the stored real scalar.

        Raises:
            Error: Negative offset or incomplete scalar.
        """
        if offset < 0 or len(raw) < self.width or offset > len(raw) - self.width:
            raise Error("incomplete DVS scalar")
        if self.kind == UInt8(98):
            return Float64(raw[offset] != 0)
        if self.width == 16:
            var scalar = List[UInt8](capacity=16)
            for i in range(16):
                scalar.append(raw[offset + i])
            return external_call["sc_dvs_extended", Float64](
                scalar.unsafe_ptr(), Int32(self.big)
            )
        var bits = UInt64(0)
        for i in range(self.width):
            var shift = i
            if self.big:
                shift = self.width - 1 - i
            bits |= UInt64(raw[offset + i]) << UInt64(shift * 8)
        if self.kind == UInt8(117):
            return Float64(bits)
        if self.kind == UInt8(105):
            var shift = 64 - self.width * 8
            return Float64(Int64(bits << UInt64(shift)) >> Int64(shift))
        if self.width == 8:
            return bitcast[DType.float64](bits)
        if self.width == 4:
            return Float64(bitcast[DType.float32](UInt32(bits)))
        return Float64(bitcast[DType.float16](UInt16(bits)))


@export
def dvs_scalar_decode_c(
    raw_addr: Int, raw_bytes: Int, offset: Int,
    kind: Int, width: Int, big: Int, output_addr: Int,
) abi("C") -> Int32:
    """Decode one caller-owned scalar atomically through the native C boundary.

    Return zero on success and -1 on invalid sizes, offsets, addresses,
    alignment, overlap or scalar declaration. Input and output must identify
    live immutable input and exclusive writable double storage respectively.
    Refusal never changes the destination. This ABI targets 64-bit Linux.

    Args:
        raw_addr: Address of immutable live input allocation.
        raw_bytes: Allocation byte length.
        offset: Scalar byte offset within the input.
        kind: ASCII scalar kind b, i, u or f.
        width: Stored scalar byte width.
        big: Byte-order flag, zero for little endian and one for big endian.
        output_addr: Address of exclusively writable aligned double storage.

    Returns:
        Zero for successful decode, minus one for refusal.

    """
    if (
        raw_addr <= 0 or output_addr <= 0 or output_addr % 8 != 0
        or raw_bytes < 0 or raw_addr > 0x7FFFFFFFFFFFFFFF - raw_bytes
        or output_addr > 0x7FFFFFFFFFFFFFFF - 8
        or offset < 0 or width <= 0 or width > raw_bytes
        or offset > raw_bytes - width or kind < 0 or kind > 255
        or (big != 0 and big != 1)
    ):
        return -1
    if raw_addr < output_addr + 8 and output_addr < raw_addr + raw_bytes:
        return -1
    try:
        var type = Scalar(UInt8(kind), width, big != 0)
        var raw = List[UInt8](capacity=width)
        var source = Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=raw_addr + offset
        )
        for i in range(width):
            raw.append(source[unsafe_offset=i])
        var value = type.decode(raw, 0)
        var destination = Pointer[Float64, MutAnyOrigin](
            unsafe_from_address=output_addr
        )
        destination[] = value
        return 0
    except:
        return -1
