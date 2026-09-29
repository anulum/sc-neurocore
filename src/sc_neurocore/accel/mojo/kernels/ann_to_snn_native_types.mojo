# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo dense IF C ABI descriptors

"""ABI-one layouts for 64-bit Linux; raw spans require live aligned caller storage."""

from std.memory import Pointer


@fieldwise_init
struct LayerSpec(Copyable, Movable):
    """C-compatible 72-byte borrowed dense coefficient and state descriptor."""
    var outputs: UInt
    """Positive output width."""
    var inputs: UInt
    """Positive input width."""
    var weights: UInt
    """Aligned live row-major coefficient address."""
    var bias: UInt
    """Aligned current address, nullable only for absent bias."""
    var bias_len: UInt
    """Zero or exactly the output width."""
    var threshold: Float64
    """Positive finite inclusive threshold."""
    var initial_fraction: Float64
    """Finite membrane preload fraction."""
    var initial: UInt
    """Aligned supplied state address, nullable for empty state."""
    var initial_len: UInt
    """Complete batch/output state element count."""


@fieldwise_init
struct ReplayRequest(Copyable, Movable):
    """C-compatible 64-byte immutable replay request, borrowed for one call."""
    var version: UInt32
    """Exact ownership ABI version one."""
    var flags: UInt32
    """Trace/binary/linear/supplied-state flags in bits zero through three."""
    var layers: UInt
    """Live aligned complete LayerSpec array address."""
    var layer_count: UInt
    """Positive number of connected weighted stages."""
    var frames: UInt
    """Live aligned time/batch/input drive address."""
    var frames_len: UInt
    """Complete frame element count."""
    var steps: UInt
    """Nonnegative explicit timestep count."""
    var batch: UInt
    """Nonnegative sample count."""
    var max_working_bytes: UInt
    """Positive shared numeric reservation limit."""


@fieldwise_init
struct BufferView(Copyable, Movable):
    """C-compatible 16-byte borrowed vector, valid until its result owner is freed."""
    var data: UInt
    """Aligned result doubles address, zero for an empty vector."""
    var length: UInt
    """Result vector element count."""


def checked_count(value: UInt) raises -> Int:
    """Require a raw unsigned count to fit signed address arithmetic.

    Args:
        value: Borrowed unsigned ABI count.

    Returns:
        Nonnegative addressable signed count.

    Raises:
        Error: Unaddressable count.
    """
    if value > UInt(0x7FFFFFFFFFFFFFFF):
        raise Error("IF invalid input")
    return Int(value)


def check_span(address: UInt, count: Int, stride: Int) raises:
    """Admit a complete aligned low-address span before constructing a raw pointer.

    Args:
        address: Caller-owned storage address.
        count: Nonnegative element count; empty storage may be null.
        stride: Positive ABI element size, with eight-byte alignment.

    Raises:
        Error: Null nonempty span, alignment or address extent refusal.
    """
    if count < 0 or count > 0x7FFFFFFFFFFFFFFF // stride:
        raise Error("IF invalid input")
    if count == 0:
        return
    if address == 0 or address % 8 != 0 or address > UInt(0x7FFFFFFFFFFFFFFF - count * stride):
        raise Error("IF invalid input")


def copy_doubles(address: UInt, count: Int) raises -> List[Float64]:
    """Snapshot one admitted raw span into exact-capacity independently owned storage.

    Args:
        address: Live immutable aligned caller doubles.
        count: Admitted nonnegative element count.

    Returns:
        Independent exact-capacity coefficient, frame or state owner.

    Raises:
        Error: Invalid raw address extent.
    """
    check_span(address, count, 8)
    var values = List[Float64](capacity=count)
    if count != 0:
        var source = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=Int(address))
        for i in range(count):
            values.append(source[unsafe_offset=i])
    return values^
