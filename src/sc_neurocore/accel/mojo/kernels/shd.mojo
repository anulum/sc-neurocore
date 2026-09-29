# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo indexed SHD recording API

"""Read actual selected auditory recordings directly through system HDF5."""

from std.collections import List
from std.memory import Pointer
from std.sys.info import size_of
from shd_hdf5 import Context


def read_row(
    mut context: Context, var path: String, index: Int, maximum: Int
) raises -> Tuple[List[Float64], Int64]:
    """Read paired numeric vectors and form owned row-major event values."""
    var id = context.library.get_function[Int64]("H5Fopen")(
        path.as_c_string_slice(), UInt32(0), Int64(0)
    )
    var file = context.own(id, "H5Fclose")
    var times = context.dataset(file, "spikes/times")
    var units = context.dataset(file, "spikes/units")
    var labels = context.dataset(file, "labels")
    context.numeric_vlen(times)
    context.numeric_vlen(units)
    var ts = context.selected_space(times, index)
    var us = context.selected_space(units, index)
    var ls = context.selected_space(labels, index)
    if ts[1] != us[1] or ts[1] != ls[1]:
        raise Error("SHD recording counts differ")
    var one = UInt64(1)
    var memory_id = context.library.get_function[Int64]("H5Screate_simple")(
        Int32(1), Pointer(to=one), Int(0)
    )
    var memory = context.own(memory_id, "H5Sclose")
    var native = context.native("H5T_NATIVE_DOUBLE_g")
    var type_id = context.library.get_function[Int64]("H5Tvlen_create")(native)
    var type = context.own(type_id, "H5Tclose")
    var tb = context.vector_bytes(times, type, ts[0])
    var ub = context.vector_bytes(units, type, us[0])
    if tb > UInt64(maximum // 2) or ub > UInt64(maximum // 2) - tb:
        raise Error("SHD recording exceeds its event budget")
    var time = context.vector(times, type, memory, ts[0], maximum)
    var channels = context.vector(units, type, memory, us[0], maximum)
    if len(time) != len(channels):
        raise Error("SHD event vectors differ")
    var label = context.label(labels, memory, ls[0])
    var events = List[Float64](capacity=len(time) * 4)
    for i in range(len(time)):
        events.append(channels[i])
        events.append(Float64(0))
        events.append(Float64(0))
        events.append(time[i] * Float64(1000))
    return (events^, label)


def read_shd_recording(
    path: String,
    index: Int,
    maximum_bytes: Int = 67108864,
    hdf5_library: String = "libhdf5_serial.so",
) raises -> Tuple[List[Float64], Int64]:
    """Read one zero-based row as owned x/y/polarity/millisecond doubles and int64 label.

    Numeric variable-length time/channel vectors and integer labels require equal
    rank-one recording counts. Widen to double before milliseconds; preserve empty
    rows. Invalid index, label overflow, budget or native library refuses without
    download or decoder substitution. The budget limits the event result, not all
    HDF5 vectors or temporary memory. The caller validates geometry/finiteness and
    manifest identity. Use an isolated process; this API does not synchronise other
    HDF5 callers. Every registered identifier and read VLEN buffer is released.
    """
    if (
        index < 0
        or maximum_bytes < 0
        or path.byte_length() == 0
        or "\0" in path
    ):
        raise Error("invalid SHD path, index or event budget")
    var context = Context(hdf5_library)
    try:
        return read_row(context, path, index, maximum_bytes)
    finally:
        context.close()
