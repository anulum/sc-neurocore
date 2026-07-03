# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Operator abstraction for tractable unbounded equivalence proofs

"""Lift an internal combinational result to a free input port (operator abstraction).

A sequential-equivalence proof over a fixed-point neuron bit-blasts every
multiplier. For k-induction (an unbounded proof) that is intractable — ``z3``
closes the LIF miter only at a narrow 4-bit datapath and stalls as the multiplier
widens; a blackbox uninterpreted function is not an option either, since yosys
0.33's ``smtbmc`` crashes on a blackbox submodule.

The tractable, *sound* alternative is to **abstract** the multiplier: remove the
signal that carries its product and expose that signal as a free input port
instead. When both the device-under-test and the reference lift the *same* product
to the *same* input name, the miter drives both instances from one shared free
wire, so the two products are equal by construction (congruence) and the solver
never reasons about multiplication at all.

The abstraction is a sound over-approximation for a **PASS**: proving the
abstracted modules equivalent for *every* value of the free product proves them
equivalent for the real product in particular, so the concrete modules are
equivalent. (A ``FAIL`` on the abstracted proof may be spurious — the abstraction
is coarser than the design — so abstraction is opt-in, not the default.)

The transform operates on a single-module source (the compiler's generated RTL
and the reference model are each one module). For each lifted signal it renames
the internal name to the target port name, drops the signal's declaration and its
continuous-assign driver, and adds the port to the module's ANSI interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .equivalence_miter import _PORT_RE, _module_header, _port_list_bounds

__all__ = ["LiftedSignal", "abstract_to_free_inputs"]


@dataclass(frozen=True)
class LiftedSignal:
    """One internal result abstracted to a free input port.

    Attributes
    ----------
    internal : str
        Name of the internal signal to abstract (e.g. a multiplier product wire).
    port : str
        Name of the free input port to expose it as. It may differ from
        ``internal`` so two modules that name the product differently can present
        the same abstracted interface for the miter to share.
    msb : str or None
        Most-significant bit-index expression of the port width, preserving a
        parameter-dependent width (``"2*DATA_WIDTH-1"``). ``None`` declares a
        scalar (1-bit) port.
    signed : bool
        Whether the port is declared ``signed``.
    """

    internal: str
    port: str
    msb: str | None = None
    signed: bool = False

    def declaration(self) -> str:
        """Return the ``input wire`` port declaration for the module header."""
        signed = " signed" if self.signed else ""
        span = f" [{self.msb}:0]" if self.msb is not None else ""
        return f"input wire{signed}{span} {self.port}"


def _drop_signal_definition(verilog: str, name: str) -> str:
    """Remove ``name``'s declaration and its continuous-assign driver.

    Handles both single-statement forms the compiler and hand-written references
    emit: an inline ``wire NAME = EXPR;`` (declaration *is* the driver), or a bare
    ``wire NAME;`` paired with a separate ``assign NAME = EXPR;``. The name is
    anchored as the declared/driven identifier (before ``=`` or ``;``), so a use of
    ``name`` in another statement's right-hand side is never matched. Raises
    :class:`ValueError` when neither form is found.
    """
    esc = re.escape(name)
    inline_re = re.compile(rf"[ \t]*(?:wire|reg)\b[^;\n]*\b{esc}\s*=[^;\n]*;\n?")
    decl_re = re.compile(rf"[ \t]*(?:wire|reg)\b[^;\n]*\b{esc}\s*;\n?")
    assign_re = re.compile(rf"[ \t]*assign\s+{esc}\s*=[^;]*;\n?")

    if inline_re.search(verilog) is not None:
        # ``wire NAME = EXPR;`` — the declaration carries the driver.
        return inline_re.sub("", verilog, count=1)
    if decl_re.search(verilog) is None:
        raise ValueError(f"no single-signal declaration found for {name!r}")
    if assign_re.search(verilog) is None:
        raise ValueError(f"no continuous-assign driver found for {name!r}")
    return assign_re.sub("", decl_re.sub("", verilog, count=1), count=1)


def abstract_to_free_inputs(verilog: str, *, top: str, signals: list[LiftedSignal]) -> str:
    """Abstract each signal's driver away and expose it as a free input port.

    For every :class:`LiftedSignal` the internal name is renamed to the target
    port name, its declaration and continuous-assign driver are removed, and the
    port is added to ``top``'s ANSI interface. The result over-approximates the
    original: the abstracted result is unconstrained, so a miter proof that
    survives it is sound for the concrete design (see the module docstring).

    Parameters
    ----------
    verilog : str
        Single-module Verilog source containing ``top``.
    top : str
        Module to transform.
    signals : list[LiftedSignal]
        The internal results to abstract. Must be non-empty with unique port
        names that do not collide with an existing port.

    Returns
    -------
    str
        The transformed Verilog source.

    Raises
    ------
    ValueError
        If ``signals`` is empty, a port name is duplicated or already declared, or
        a signal's declaration / driver cannot be located.
    """
    if not signals:
        raise ValueError("signals must not be empty")

    ports = [s.port for s in signals]
    if len(set(ports)) != len(ports):
        raise ValueError("lifted port names must be unique")

    existing = {m.group("name") for m in _PORT_RE.finditer(_module_header(verilog, top))}
    for name in ports:
        if name in existing:
            raise ValueError(f"lifted port {name!r} already exists in module {top!r}")

    out = verilog
    for signal in signals:
        if signal.port != signal.internal:
            out = re.sub(rf"\b{re.escape(signal.internal)}\b", signal.port, out)
        out = _drop_signal_definition(out, signal.port)

    open_idx, close_idx = _port_list_bounds(out, top)
    decls = ",\n    " + ",\n    ".join(s.declaration() for s in signals)
    return out[:close_idx] + decls + "\n" + out[close_idx:]
