# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit equation word arithmetic

"""Represent equation widths, literal encoding and product rounding in native IR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..verilog_compiler_config import Q88

if TYPE_CHECKING:
    from sc_neurocore_engine.ir import ScGraphBuilder


class EquationArithmetic:
    """Append raw signed words under the existing equation compiler's Q contract.

    Parameters
    ----------
    builder : ScGraphBuilder
        Destination native graph; all methods append real typed instructions.
    q : Q88
        Word geometry and product-rounding policy. Products wrap before
        rounding, exactly as the maintained Verilog intermediate wires do.
    """

    def __init__(self, builder: ScGraphBuilder, q: Q88) -> None:
        self.builder = builder
        self.q = q
        self.constants: dict[tuple[int, int], int] = {}

    def constant(self, value: int, width: int) -> int:
        """Append a width-wrapped constant, including words wider than 64 bits.

        Parameters
        ----------
        value : int
            Raw integer, before two's-complement wrapping.
        width : int
            Positive physical width.

        Returns
        -------
        int
            Native signed-word SSA identifier. Wide values are composed from
            30-bit limbs through actual integer shifts and additions.
        """
        value &= (1 << width) - 1
        if value >= 1 << (width - 1):
            value -= 1 << width
        key = (value, width)
        if key in self.constants:
            return self.constants[key]
        b, ty = self.builder, f"i{width}"
        if -(1 << 63) <= value < 1 << 63:
            result = (
                b.word_resize(self.constant(value, 64), ty)
                if width > 64
                else b.constant_i64(value, ty)
            )
        else:
            magnitude = abs(value)
            limbs = []
            while magnitude:
                limbs.append(magnitude & ((1 << 30) - 1))
                magnitude >>= 30
            result = self.constant(limbs.pop(), width)
            for limb in reversed(limbs):
                result = b.word_binary("shl", result, self.constant(30, width), ty)
                result = b.word_binary("add", result, self.constant(limb, width), ty)
            if value < 0:
                result = b.word_unary("neg", result, ty)
        self.constants[key] = result
        return result

    def literal(self, value: float, width: int) -> int:
        """Encode a source Q-format literal and extend it to its expression context.

        Parameters
        ----------
        value : float
            Source real number, quantised by the existing Q encoder.
        width : int
            Destination expression width.

        Returns
        -------
        int
            Native signed-word SSA identifier, without intermediate saturation.
        """
        dw = self.q.data_width
        raw = self.q.encode(value)
        if raw >= 1 << (dw - 1):
            raw -= 1 << dw
        return self.constant(raw, width)

    def round_product(self, value: int, width: int) -> int:
        """Narrow one wide fixed-point product using the maintained bias arithmetic.

        Parameters
        ----------
        value : int
            SSA signed word of the given width, already modulo-width wrapped.
        width : int
            Product width, normally twice the Q data width.

        Returns
        -------
        int
            SSA signed word of Q data width.

        Raises
        ------
        ValueError
            If the rounding policy is not truncate, nearest or bankers.
        """
        b, q, ty = self.builder, self.q, f"i{width}"
        if q.rounding not in {"truncate", "nearest", "bankers"}:
            raise ValueError(
                "native equation products require truncate, nearest or bankers rounding"
            )
        if q.fraction == 0:
            return b.word_resize(value, f"i{q.data_width}")
        half = 1 << (q.fraction - 1)
        rounded = value
        if q.rounding != "truncate":
            bias = self.constant(half, width)
            if q.rounding == "nearest":
                negative = b.word_compare("lt", value, self.constant(0, width))
                bias = b.word_select(negative, self.constant(half - 1, width), bias, ty)
            rounded = b.word_binary("add", value, bias, ty)
        rounded = b.word_binary("ashr", rounded, self.constant(q.fraction, width), ty)
        if q.rounding == "bankers":
            residue = b.word_binary("and", value, self.constant((1 << q.fraction) - 1, width), ty)
            tie = b.word_compare("eq", residue, self.constant(half, width))
            mask = b.word_select(tie, self.constant(-2, width), self.constant(-1, width), ty)
            rounded = b.word_binary("and", rounded, mask, ty)
        return b.word_resize(rounded, f"i{q.data_width}")
