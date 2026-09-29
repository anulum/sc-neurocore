# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native converted DVS NPY recordings

"""Read real NPY camera recordings into owned row-major native doubles."""

from std.collections import List
from std.collections.string import chr
from std.ffi import external_call
from std.memory import Pointer
from dvs_header import parse_header


def _exact(fd: Int32, count: Int) raises -> List[UInt8]:
    var result = List[UInt8](unsafe_uninit_length=count)
    var offset = 0
    while offset < count:
        var received = external_call["read", Int](Int(fd), result.unsafe_ptr().unsafe_offset(offset), count - offset)
        if received <= 0:
            raise Error("incomplete DVS NPY recording")
        offset += received
    return result^


def _read(fd: Int32, maximum: Int) raises -> List[Float64]:
    var prefix = _exact(fd, 8)
    var magic: List[UInt8] = [0x93, 78, 85, 77, 80, 89]
    for i in range(6):
        if prefix[i] != magic[i]:
            raise Error("invalid DVS NPY preamble")
    var version = Int(prefix[6])
    if version < 1 or version > 3 or prefix[7] != 0:
        raise Error("unsupported DVS NPY format version")
    var width = 4
    if version == 1:
        width = 2
    var size = _exact(fd, width)
    var length = UInt32(0)
    for i in range(width):
        length |= UInt32(size[i]) << UInt32(i * 8)
    if length == 0 or length > 10000:
        raise Error("invalid DVS header length")
    var raw = _exact(fd, Int(length))
    if raw[len(raw) - 1] != 10:
        raise Error("DVS NPY header must end with a newline")
    var text = String()
    if version == 3:
        text = String(from_utf8=raw)
    else:
        for byte in raw:
            text += chr(Int(byte))
    var header = parse_header(text)
    if header.rows > maximum // 32:
        raise Error("DVS recording exceeds its event budget")
    var count = header.rows * 4
    if count > 0x7FFFFFFFFFFFFFFF // header.scalar.width:
        raise Error("DVS source payload exceeds native size")
    var payload = _exact(fd, count * header.scalar.width)
    var extra = UInt8(0)
    if external_call["read", Int](Int(fd), Pointer(to=extra), Int(1)) != 0:
        raise Error("DVS recording contains extra content")
    var events = List[Float64](unsafe_uninit_length=count)
    for i in range(count):
        var target = i
        if header.fortran:
            target = (i % header.rows) * 4 + i // header.rows
        events[target] = header.scalar.decode(payload, i * header.scalar.width)
    return events^


def read_dvs_recording(path: String, maximum_bytes: Int = 67108864) raises -> List[Float64]:
    """Read one bounded numeric NPY camera recording without Python or pickle.

    NPY versions 1/2/3, C/Fortran layouts, either byte order and real scalar
    aliases preserve the stored values through float64 widening. Four columns
    represent x, y, polarity and millisecond timestamps. The returned-byte
    budget is checked before payload allocation; input and temporary memory
    are additional. Geometry, finiteness, ordering and manifest identity belong
    to the dataset caller. Every opened descriptor closes on success or refusal.

    Args:
        path: Local NPY recording; no download or discovery occurs.
        maximum_bytes: Nonnegative returned float64 matrix byte limit.

    Returns:
        Owned flat row-major doubles with four values per event.

    Raises:
        Error: Invalid path, metadata, datatype, budget, truncation or extra data.
    """
    if maximum_bytes < 0 or path.byte_length() == 0 or "\0" in path:
        raise Error("invalid DVS path or event budget")
    var name = path
    var fd = external_call["open", Int32](name.as_c_string_slice(), Int32(0))
    if fd < 0:
        raise Error("DVS recording is unreadable")
    try:
        return _read(fd, maximum_bytes)
    finally:
        if external_call["close", Int32](fd) != 0:
            raise Error("DVS descriptor close failed")
