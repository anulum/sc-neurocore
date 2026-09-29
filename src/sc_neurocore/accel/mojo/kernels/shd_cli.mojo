# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Supervised Mojo SHD binary worker

"""Read one native auditory recording with bounded process lifetime and binary output."""

from shd import read_shd_recording
from std.ffi import external_call
from std.memory import Pointer
from std.sys import argv
from std.sys.info import size_of
from std.sys.terminate import exit


def write_value[T: Copyable & Deinitable](var value: T) raises:
    """Write one protocol scalar exactly, refusing an incomplete system write.
    """
    var size = size_of[T]()
    if external_call["write", Int](Int(1), Pointer(to=value), size) != size:
        raise Error("SHD binary write failed")


def run() raises:
    """Supervise the actual native read and emit only its completed recording.
    """
    var args = argv()
    if len(args) != 6:
        raise Error("path index budget HDF5-library parent required")
    if (
        external_call["prctl", Int32](
            Int32(1), UInt64(9), UInt64(0), UInt64(0), UInt64(0)
        )
        != 0
    ):
        raise Error("SHD parent-death signal setup failed")
    if Int(external_call["getppid", Int32]()) != Int(args[5]):
        exit(124)
    _ = external_call["alarm", UInt32](UInt32(30))
    var sample = read_shd_recording(
        args[1], Int(args[2]), Int(args[3]), args[4]
    )
    write_value(UInt32(0x31444853))
    write_value(UInt64(len(sample[0])))
    write_value(sample[1])
    for value in sample[0]:
        write_value(value)


def main():
    """Keep error text out of the binary protocol and exit with a refusal status.
    """
    try:
        run()
    except:
        exit(1)
