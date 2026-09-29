# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NPY metadata shape and layout admission

"""Validate the exact three NPY metadata fields before payload allocation."""

from dvs_literals import Parser
from dvs_dtype import parse_dtype
from dvs_scalar import Scalar
from dvs_strings import byte_at


@fieldwise_init
struct Header(Copyable, Movable):
    """Describe admitted rank-two real storage with four camera-event columns."""
    var rows: Int
    """Nonnegative number of stored camera events."""
    var fortran: Bool
    """Whether the source matrix is stored in column-major order."""
    var scalar: Scalar
    """Validated stored numeric type and byte order."""


def parse_header(text: String) raises -> Header:
    """Require one inert dictionary, unique fields and a nonnegative row count.

    Args:
        text: Decoded NPY metadata dictionary with final newline.

    Returns:
        Four-column recording shape, storage order and real scalar type.

    Raises:
        Error: Malformed, duplicate, missing or extra fields, invalid shape or type.
    """
    if "\0" in text:
        raise Error("NUL in DVS header")
    var parser = Parser(text.replace("\r\n", "\n").replace("\r", "\n"))
    parser.skip()
    var line = parser.position - 1
    while line >= 0 and byte_at(parser.text, line) != 10 and byte_at(parser.text, line) != 12:
        line -= 1
    var indentation = line + 1 < parser.position
    for i in range(line + 1, parser.position):
        var ch = byte_at(parser.text, i)
        if ch != 32 and ch != 9:
            indentation = False
    if indentation:
        raise Error("indented DVS header")
    var root = parser.value()
    parser.skip()
    if parser.position != parser.text.byte_length():
        raise Error("trailing DVS expression")
    var fields = parser.nodes[root].copy()
    if fields.kind != 5 or len(fields.keys) != 3:
        raise Error("DVS header must have three dictionary fields")
    var descriptor = String()
    var order = -1
    var rows = -1
    for i in range(len(fields.keys)):
        var key = fields.keys[i]
        var value = parser.nodes[fields.children[i]].copy()
        if key == "descr" and value.kind == 1:
            descriptor = value.text
        elif key == "fortran_order" and value.kind == 2:
            order = Int(value.number)
        elif key == "shape" and value.kind == 4 and len(value.children) == 2:
            var row = parser.nodes[value.children[0]].copy()
            var column = parser.nodes[value.children[1]].copy()
            if row.kind != 3 or row.overflow or (row.negative and row.number != 0) or row.number > UInt64(0x7FFFFFFFFFFFFFFF):
                raise Error("invalid DVS rows")
            if column.kind != 3 or column.overflow or column.negative or column.number != 4:
                raise Error("DVS requires four columns")
            rows = Int(row.number)
        else:
            raise Error("invalid DVS metadata field")
    if order < 0 or rows < 0 or descriptor.byte_length() == 0:
        raise Error("missing DVS metadata field")
    return Header(rows, order != 0, parse_dtype(descriptor))
