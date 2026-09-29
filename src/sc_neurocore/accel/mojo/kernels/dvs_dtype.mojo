# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NPY scalar descriptor admission

"""Resolve NumPy real scalar aliases for native DVS input conversion."""

from std.collections import List
from std.collections.string import chr
from dvs_scalar import Scalar
from dvs_strings import byte_at


def parse_dtype(source: String) raises -> Scalar:
    """Resolve an admitted real descriptor without discovery or coercion.

    Args:
        source: Decoded NPY real scalar descriptor.

    Returns:
        Validated storage kind, byte width and byte order.

    Raises:
        Error: Unknown, nonreal, incompatible or prefixed named scalar type.
    """
    var prefix = byte_at(source, 0)
    var explicit = prefix == 60 or prefix == 62 or prefix == 61 or prefix == 124
    var word = source
    if explicit:
        word = String(source[byte=1:])
    else:
        prefix = 61
    var names: List[String] = ["bool", "bool_", "byte", "ubyte", "short", "ushort", "intc", "int32", "uintc", "uint32", "int8", "uint8", "int16", "uint16", "int64", "longlong", "uint64", "ulonglong", "half", "float16", "single", "float32", "double", "float64", "float", "longdouble", "float128", "int", "int_", "intp", "uint", "uintp", "long", "ulong"]
    var resolved: List[String] = ["b1", "b1", "i1", "u1", "i2", "u2", "i4", "i4", "u4", "u4", "i1", "u1", "i2", "u2", "i8", "i8", "u8", "u8", "f2", "f2", "f4", "f4", "f8", "f8", "f8", "f16", "f16", "i8", "i8", "i8", "u8", "u8", "i8", "u8"]
    for i in range(len(names)):
        if word == names[i]:
            if explicit:
                raise Error("prefixed named DVS dtype")
            word = resolved[i]
            break
    var characters: List[String] = ["?", "b", "B", "h", "H", "i", "I", "q", "Q", "e", "f", "d", "g", "p", "n", "P", "N", "l", "L"]
    var aliases: List[String] = ["b1", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "f16", "i8", "i8", "u8", "u8", "i8", "u8"]
    for i in range(len(characters)):
        if word == characters[i]:
            word = aliases[i]
            break
    var numeric: List[String] = ["b1", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "i8", "u8", "f4", "f8", "f16"]
    for i in range(len(numeric)):
        if word == chr(i):
            word = numeric[i]
            break
    if word == chr(23):
        word = "f2"
    var kind = byte_at(word, 0)
    var cursor = 1
    if byte_at(word, cursor) == 43:
        cursor += 1
    if cursor >= word.byte_length():
        raise Error("invalid DVS scalar width")
    var width = Int(0)
    while cursor < word.byte_length():
        var ch = byte_at(word, cursor)
        if ch < 48 or ch > 57 or width > 16:
            raise Error("invalid DVS scalar width")
        width = width * 10 + ch - 48
        cursor += 1
    return Scalar(UInt8(kind), width, prefix == 62)
