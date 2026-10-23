# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation timestep clock control

"""Represent the registered equation compiler's pipeline and macro-step cadence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sc_neurocore_engine.ir import ScGraphBuilder


@dataclass(frozen=True)
class EquationClock:
    """Native conditions controlling one equation's state and random sequence.

    Attributes
    ----------
    advance : int
        Boolean SSA value enabling state commits on the coming clock edge.
    macro_boundary : int
        Boolean SSA value enabling macro history, threshold and RNG commits.
    clocks_per_timestep : int
        Number of physical clocks per complete model timestep. Pipeline storage
        keeps capturing on every clock; it must not use either enable.
    """

    advance: int
    macro_boundary: int
    clocks_per_timestep: int


def build_clock(
    builder: ScGraphBuilder, *, pipeline_registers: int = 0, substeps: int = 1
) -> EquationClock:
    """Append the exact pipeline fill or macro-step counter to a native graph.

    Parameters
    ----------
    builder : ScGraphBuilder
        Destination graph. Counter storage has asynchronous active-low zero reset.
    pipeline_registers : int, optional
        Actual selected product-register count, rather than requested stage count
        or longest path. State commits once every count plus one clocks, matching
        the maintained registered Verilog compiler's conservative fill counter.
    substeps : int, optional
        Existing EquationNeuron substep count. Without pipelining, state advances
        every clock and macro history/RNG advances on the last substep.

    Returns
    -------
    EquationClock
        Explicit native Bool conditions and the complete timestep period.

    Raises
    ------
    ValueError
        Counts are not integers in range, need more than 64 counter bits, or
        request pipelining together with macro substeps. Admission occurs before
        any graph mutation, as for the maintained compiler's rejected combination.

    Notes
    -----
    `_verilog_registered_module.py` resets the fill count after every valid
    recurrence, not just after reset. Its count is the number of physical product
    registers, which can exceed the actual dependency depth.
    """
    if type(pipeline_registers) is not int or pipeline_registers < 0:
        raise ValueError("pipeline register count must be a non-negative integer")
    if type(substeps) is not int or substeps < 1:
        raise ValueError("equation substeps must be a positive integer")
    if pipeline_registers and substeps > 1:
        raise ValueError("equation pipelining and macro substeps cannot be combined")
    limit = pipeline_registers if pipeline_registers else substeps - 1
    if limit >= 1 << 64:
        raise ValueError("equation timestep counter requires more than 64 bits")
    zero = builder.constant_i64(0, "i1")
    every_clock = builder.word_compare("eq", zero, zero)
    if limit == 0:
        return EquationClock(every_clock, every_clock, 1)

    width = limit.bit_length()
    ty = f"u{width}"

    def unsigned(value: int) -> int:
        """Preserve the full counter word through the signed i64 literal API."""
        signed = value - (1 << width) if value >= 1 << (width - 1) else value
        return builder.word_resize(builder.constant_i64(signed, f"i{width}"), ty)

    reset = builder.constant_i64(0, ty)
    counter = builder.state(reset, ty)
    boundary = builder.word_compare("eq", counter, unsigned(limit))
    increment = builder.word_binary("add", counter, unsigned(1), ty)
    builder.store(counter, builder.word_select(boundary, reset, increment, ty))
    return EquationClock(boundary if pipeline_registers else every_clock, boundary, limit + 1)
