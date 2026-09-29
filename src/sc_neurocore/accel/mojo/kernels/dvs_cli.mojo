# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Supervised Mojo DVS binary command

"""Read one native camera recording under Linux parent and deadline guards."""

from std.collections import List
from std.ffi import external_call
from std.memory import Pointer
from std.sys import argv
from std.sys.terminate import exit
from dvs_recordings import read_dvs_recording


def _write_all[T: Copyable & Deinitable, origin: Origin](data: Pointer[T, origin], size: Int) raises:
    var bytes = data.unsafe_bitcast[UInt8]()
    var offset = 0
    while offset < size:
        var written = external_call["write", Int](Int(1), bytes.unsafe_offset(offset), size - offset)
        if written <= 0:
            raise Error("DVS binary write failed")
        offset += written


def run() raises:
    """Bind lifetime before input access, then emit only a complete decoded array.

    Raises:
        Error: Invalid arguments, missing parent, guard failure, invalid recording or output write failure.
    """
    var args = argv()
    if len(args) != 3 and len(args) != 4:
        raise Error("path budget [expected-parent] required")
    var budget = Int(args[2])
    var parent = external_call["getppid", Int32]()
    var expected = Int(parent)
    if len(args) == 4:
        expected = Int(args[3])
    if expected <= 0 or expected != Int(parent):
        raise Error("DVS expected parent absent")
    if external_call["prctl", Int32](Int32(1), UInt64(9), UInt64(0), UInt64(0), UInt64(0)) != 0:
        raise Error("DVS parent guard refused")
    if Int(external_call["getppid", Int32]()) != expected:
        exit(124)
    _ = external_call["alarm", UInt32](UInt32(30))
    var events = read_dvs_recording(args[1], budget)
    var magic = UInt32(0x31535644)
    var count = UInt64(len(events))
    _write_all(Pointer(to=magic), 4)
    _write_all(Pointer(to=count), 8)
    if len(events) != 0:
        _write_all(events.unsafe_ptr(), len(events) * 8)


def main():
    """Keep diagnostics out of the binary stream and refuse with a nonzero exit."""
    try:
        run()
    except:
        exit(1)
