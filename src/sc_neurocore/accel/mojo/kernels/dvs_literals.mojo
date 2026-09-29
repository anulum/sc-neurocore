# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Inert NPY literal structure

"""Parse bounded metadata literals into an arena without running Python."""

from std.collections import List
from dvs_strings import byte_at, whitespace, string_start, parse_strings


@fieldwise_init
struct Literal(Copyable, Movable):
    """Store a scalar or container with arena indices for its children."""
    var kind: Int
    """Node category: string=1, Boolean=2, integer=3, tuple=4, dictionary=5."""
    var text: String
    """Decoded string or scalar token spelling."""
    var number: UInt64
    """Unsigned integer magnitude or Boolean value."""
    var overflow: Bool
    """Whether the integer magnitude exceeds uint64."""
    var negative: Bool
    """Whether one unary minus was applied."""
    var signed: Bool
    """Whether one unary operator was applied."""
    var children: List[Int]
    """Stable arena indices for tuple elements or dictionary values."""
    var keys: List[String]
    """Decoded unique dictionary field names."""


struct Parser(Movable):
    """Own a bounded header and all decoded literal nodes."""
    var text: String
    """Header input with normalised line endings."""
    var position: Int
    """Current zero-based byte position in the input."""
    var depth: Int
    """Active container depth, limited to 200."""
    var nodes: List[Literal]
    """Owned decoded literal arena."""

    def __init__(out self, text: String):
        """Reserve only the header parser state; container depth is checked.

        Args:
            text: Header source with normalised newlines.
        """
        self.text = text
        self.position = 0
        self.depth = 0
        self.nodes = List[Literal]()

    def skip(mut self):
        """Advance past Python whitespace and comments."""
        self.position = whitespace(self.text, self.position)

    def consume(mut self, token: Int) -> Bool:
        """Consume an expected punctuation byte after whitespace.

        Args:
            token: Expected ASCII punctuation byte.

        Returns:
            Whether the token was found and consumed.
        """
        self.skip()
        if byte_at(self.text, self.position) != token:
            return False
        self.position += 1
        return True

    def add(mut self, var node: Literal) -> Int:
        """Append an owned node and return its stable arena index.

        Args:
            node: Owned parsed literal with stable child indices.

        Returns:
            The appended node index.
        """
        var index = len(self.nodes)
        self.nodes.append(node^)
        return index

    def value(mut self) raises -> Int:
        """Read one inert scalar or structurally bounded container.

        Returns:
            Arena index of the parsed scalar or container.

        Raises:
            Error: Invalid token, malformed literal or nesting beyond 200 containers.
        """
        self.skip()
        var token = byte_at(self.text, self.position)
        if string_start(self.text, self.position):
            var pair = parse_strings(self.text, self.position)
            self.position = pair[1]
            return self.add(Literal(1, pair[0], 0, False, False, False, List[Int](), List[String]()))
        if token == 40 or token == 123:
            self.depth += 1
            if self.depth > 200:
                raise Error("DVS nesting exceeds Python parser limit")
            var result = self.container(token)
            self.depth -= 1
            return result
        var sign = 0
        if token == 43 or token == 45:
            sign = token
            self.position += 1
            self.skip()
        if sign != 0 and byte_at(self.text, self.position) == 40:
            var operand = self.value()
            var node = self.nodes[operand].copy()
            if node.kind != 3 or node.negative or node.signed:
                raise Error("invalid DVS unary operand")
            node.negative = sign == 45
            node.signed = True
            return self.add(node^)
        var start = self.position
        while True:
            var ch = byte_at(self.text, self.position)
            if not ((ch >= 48 and ch <= 57) or (ch >= 65 and ch <= 90) or (ch >= 97 and ch <= 122) or ch == 95):
                break
            self.position += 1
        var word = String(self.text[byte=start:self.position])
        if sign == 0 and (word == "True" or word == "False"):
            return self.add(Literal(2, word, UInt64(word == "True"), False, False, False, List[Int](), List[String]()))
        var pair = integer(word)
        return self.add(Literal(3, word, pair[0], pair[1], sign == 45, sign != 0, List[Int](), List[String]()))

    def container(mut self, token: Int) raises -> Int:
        """Read dictionary fields or tuple/group children with exact delimiters.

        Args:
            token: Opening parenthesis or dictionary brace byte.

        Returns:
            Arena index of the parsed tuple, dictionary or grouped scalar.

        Raises:
            Error: Missing punctuation, duplicate keys or a nonstring dictionary key.
        """
        self.position += 1
        var node = Literal(4, "", 0, False, False, False, List[Int](), List[String]())
        var closer = 41
        if token == 123:
            node.kind = 5
            closer = 125
        if self.consume(closer):
            return self.add(node^)
        while True:
            if token == 123:
                var key_index = self.value()
                var key = self.nodes[key_index].copy()
                if key.kind != 1:
                    raise Error("nonstring DVS dictionary key")
                for name in node.keys:
                    if name == key.text:
                        raise Error("duplicate DVS dictionary key")
                node.keys.append(key.text)
                if not self.consume(58):
                    raise Error("missing DVS field colon")
            var child = self.value()
            node.children.append(child)
            if self.consume(closer):
                if token == 40 and len(node.children) == 1:
                    return child
                break
            if not self.consume(44):
                raise Error("missing DVS literal separator")
            if self.consume(closer):
                break
        return self.add(node^)


def integer(word: String) raises -> Tuple[UInt64, Bool]:
    """Validate Python integer spelling before checking native size overflow.

    Args:
        word: Inert Python integer token, without a unary sign.

    Returns:
        Unsigned value and a flag indicating uint64 overflow.

    Raises:
        Error: Invalid radix, underscores, digits, leading zero or decimal length.
    """
    var radix = 10
    var start = 0
    if word.startswith("0x") or word.startswith("0X"):
        radix = 16
        start = 2
    elif word.startswith("0o") or word.startswith("0O"):
        radix = 8
        start = 2
    elif word.startswith("0b") or word.startswith("0B"):
        radix = 2
        start = 2
    if word.byte_length() == start:
        raise Error("empty DVS integer")
    var count = 0
    var underscore = False
    var nonzero = False
    var overflow = False
    var value = UInt64(0)
    for i in range(start, word.byte_length()):
        var ch = byte_at(word, i)
        if ch == 95:
            if underscore or (i == start and start == 0):
                raise Error("invalid DVS integer underscore")
            underscore = True
            continue
        var digit = -1
        if ch >= 48 and ch <= 57:
            digit = ch - 48
        elif ch >= 65 and ch <= 70:
            digit = ch - 55
        elif ch >= 97 and ch <= 102:
            digit = ch - 87
        if digit < 0 or digit >= radix:
            raise Error("invalid DVS integer digit")
        nonzero = nonzero or digit != 0
        count += 1
        underscore = False
        if value > (UInt64(0xFFFFFFFFFFFFFFFF) - UInt64(digit)) // UInt64(radix):
            overflow = True
        if not overflow:
            value = value * UInt64(radix) + UInt64(digit)
    if underscore or count == 0 or (start == 0 and (count > 4300 or (byte_at(word, 0) == 48 and nonzero))):
        raise Error("invalid DVS integer spelling")
    return (value, overflow)
