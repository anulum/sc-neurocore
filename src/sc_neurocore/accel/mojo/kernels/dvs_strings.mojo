# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Inert NPY string literals

"""Decode inert Python string tokens needed by NPY metadata."""

from std.collections import List
from std.collections.string import chr


def byte_at(text: String, position: Int) -> Int:
    """Return one byte or minus one at the input boundary.

    Args:
        text: Header text.
        position: Zero-based byte offset.

    Returns:
        Byte value or minus one outside the input.
    """
    if position < 0 or position >= text.byte_length():
        return -1
    return Int(text.as_bytes()[position])


def whitespace(text: String, position: Int) -> Int:
    """Skip Python whitespace, comments and explicit line continuations.

    Args:
        text: Header text.
        position: Initial byte offset.

    Returns:
        First byte after whitespace, comments and line continuations.
    """
    var cursor = position
    while True:
        var ch = byte_at(text, cursor)
        if ch == 32 or ch == 9 or ch == 10 or ch == 13 or ch == 12:
            cursor += 1
        elif ch == 35:
            while byte_at(text, cursor) != -1 and byte_at(text, cursor) != 10:
                cursor += 1
        elif ch == 92 and byte_at(text, cursor + 1) == 10:
            cursor += 2
        else:
            return cursor


def string_start(text: String, position: Int) -> Bool:
    """Recognise a quoted token with an optional raw or Unicode prefix.

    Args:
        text: Header text.
        position: Initial token byte offset.

    Returns:
        Whether a quoted raw, Unicode or unprefixed token starts here.
    """
    var first = byte_at(text, position)
    if first == 39 or first == 34:
        return True
    var next = byte_at(text, position + 1)
    return (first == 114 or first == 82 or first == 117 or first == 85) and (next == 39 or next == 34)


def _digit(value: Int) -> Int:
    if value >= 48 and value <= 57:
        return value - 48
    if value >= 65 and value <= 70:
        return value - 55
    if value >= 97 and value <= 102:
        return value - 87
    return -1


def _named(name: String) raises -> String:
    var upper = name.upper()
    var symbols: List[String] = ["LESS-THAN SIGN", "GREATER-THAN SIGN", "EQUALS SIGN", "VERTICAL LINE", "PLUS SIGN", "HYPHEN-MINUS", "QUESTION MARK", "LOW LINE"]
    var characters: List[Int] = [60, 62, 61, 124, 43, 45, 63, 95]
    for i in range(8):
        if upper == symbols[i]:
            return chr(characters[i])
    var lower = "LATIN SMALL LETTER "
    var capital = "LATIN CAPITAL LETTER "
    if upper.startswith(lower) or upper.startswith(capital):
        var start = lower.byte_length()
        if upper.startswith(capital):
            start = capital.byte_length()
        var letter = byte_at(upper, start)
        if upper.byte_length() == start + 1 and letter >= 65 and letter <= 90:
            if upper.startswith(lower):
                letter += 32
            return chr(letter)
    var digits: List[String] = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    for i in range(10):
        if upper == "DIGIT " + digits[i]:
            return chr(48 + i)
    raise Error("unsupported DVS named escape")


def parse_strings(text: String, position: Int) raises -> Tuple[String, Int]:
    """Join adjacent inert quoted literals without evaluating any expression.

    Args:
        text: Header text containing inert quoted tokens.
        position: First token byte offset.

    Returns:
        Decoded adjacent strings and the first byte after them.

    Raises:
        Error: Unterminated quotes, invalid escapes or unsupported Unicode scalar.
    """
    var result = String()
    var cursor = whitespace(text, position)
    while string_start(text, cursor):
        var first = byte_at(text, cursor)
        var raw = first == 114 or first == 82
        if first == 114 or first == 82 or first == 117 or first == 85:
            cursor += 1
        var quote = byte_at(text, cursor)
        var width = 1
        if byte_at(text, cursor + 1) == quote and byte_at(text, cursor + 2) == quote:
            width = 3
        cursor += width
        var segment = cursor
        while True:
            var ch = byte_at(text, cursor)
            if ch == -1:
                raise Error("unclosed DVS string")
            if ch == quote and (width == 1 or (byte_at(text, cursor + 1) == quote and byte_at(text, cursor + 2) == quote)):
                result += String(text[byte=segment:cursor])
                cursor += width
                break
            if ch == 10 and width == 1:
                raise Error("newline in DVS string")
            if ch != 92:
                cursor += 1
                continue
            result += String(text[byte=segment:cursor])
            cursor += 1
            var escaped = byte_at(text, cursor)
            if escaped == -1:
                raise Error("unterminated DVS escape")
            cursor += 1
            if escaped >= 128:
                var start = cursor - 1
                while byte_at(text, cursor) >= 128 and byte_at(text, cursor) < 192:
                    cursor += 1
                result += "\\" + String(text[byte=start:cursor])
                segment = cursor
                continue
            if raw:
                result += "\\" + chr(escaped)
            elif escaped == 10:
                pass
            elif escaped == 92 or escaped == 39 or escaped == 34:
                result += chr(escaped)
            elif escaped == 97 or escaped == 98 or escaped == 102 or escaped == 110 or escaped == 114 or escaped == 116 or escaped == 118:
                var code = 7
                if escaped == 98: code = 8
                elif escaped == 102: code = 12
                elif escaped == 110: code = 10
                elif escaped == 114: code = 13
                elif escaped == 116: code = 9
                elif escaped == 118: code = 11
                result += chr(code)
            elif escaped == 120 or escaped == 117 or escaped == 85:
                var digits = 2
                if escaped == 117: digits = 4
                elif escaped == 85: digits = 8
                var value = Int(0)
                for _ in range(digits):
                    var digit = _digit(byte_at(text, cursor))
                    if digit < 0:
                        raise Error("invalid DVS hexadecimal escape")
                    value = value * 16 + digit
                    cursor += 1
                if value > 0x10FFFF or (value >= 0xD800 and value <= 0xDFFF):
                    raise Error("unsupported DVS Unicode scalar")
                result += chr(value)
            elif escaped == 78:
                if byte_at(text, cursor) != 123:
                    raise Error("invalid DVS named escape")
                cursor += 1
                var start = cursor
                while byte_at(text, cursor) != 125:
                    if byte_at(text, cursor) == -1:
                        raise Error("unclosed DVS named escape")
                    cursor += 1
                result += _named(String(text[byte=start:cursor]))
                cursor += 1
            elif escaped >= 48 and escaped <= 55:
                var value = escaped - 48
                for _ in range(2):
                    var digit = byte_at(text, cursor)
                    if digit < 48 or digit > 55:
                        break
                    value = value * 8 + digit - 48
                    cursor += 1
                result += chr(value)
            else:
                result += "\\" + chr(escaped)
            segment = cursor
        cursor = whitespace(text, cursor)
    return (result^, cursor)
