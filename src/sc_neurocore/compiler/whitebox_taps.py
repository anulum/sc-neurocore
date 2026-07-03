# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Whitebox state taps for unbounded equivalence proofs

"""Expose a module's internal signals as observation output ports.

A sequential-equivalence miter proves two modules agree on their *outputs*.
Bounded model checking establishes that to a depth; an *unbounded* proof by
k-induction needs the internal states to match too — the reachable-state
invariant — or the induction step diverges from unreachable start states where
the outputs happen to agree but the hidden state does not.

The obvious way to assert that invariant is a hierarchical reference into each
instance (``dut.v_reg == ref.v``). yosys 0.33 does not resolve hierarchical
references (it parses ``dut.v_reg`` as one escaped, undriven identifier) and
silently ignores SystemVerilog ``bind``, so neither route reaches an instance's
internal state. The working alternative is to **expose** that state: instrument
each module with continuous-assign *taps* that surface the relevant registers as
extra output ports. The miter then compares those taps like any other output, and
the tap equality *is* the state-matching invariant that makes ``mode="prove"``
converge to an unbounded proof.

Taps are pure observation — a tap adds an ``output wire`` and one continuous
``assign``; it introduces no register and rewrites no existing logic, so the
instrumented module is behaviourally identical to the original on its original
ports. A tap whose source is a constant (e.g. a counter a reference model does
not have, pinned to ``0``) lets two structurally different modules present the
same tap interface so the miter can still compare them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .equivalence_miter import _PORT_RE, _module_header, _port_list_bounds

__all__ = ["StateTap", "expose_state_taps"]


@dataclass(frozen=True)
class StateTap:
    """One observation tap exposing an internal signal as an output port.

    Attributes
    ----------
    port : str
        Name of the new ``output wire`` port.
    source : str
        Verilog expression assigned to the port — typically an internal register
        name (``"v_reg"``) or a constant (``"32'd0"``) that pins a tap a peer
        module lacks.
    msb : str or None
        The most-significant bit-index expression of the port width, so a
        parameter-dependent width is preserved (``"DATA_WIDTH-1"`` renders
        ``[DATA_WIDTH-1:0]``). ``None`` declares a scalar (1-bit) port.
    signed : bool
        Whether the port is declared ``signed``.
    """

    port: str
    source: str
    msb: str | None = None
    signed: bool = False

    def declaration(self) -> str:
        """Return the ``output wire`` port declaration for the module header."""
        signed = " signed" if self.signed else ""
        span = f" [{self.msb}:0]" if self.msb is not None else ""
        return f"output wire{signed}{span} {self.port}"

    def assignment(self) -> str:
        """Return the continuous ``assign`` that drives the tap from its source."""
        return f"    assign {self.port} = {self.source};"


def expose_state_taps(verilog: str, *, top: str, taps: list[StateTap]) -> str:
    """Instrument ``top`` to expose internal signals as observation output ports.

    Adds each tap's ``output wire`` port to the module's ANSI port list and a
    continuous ``assign`` before ``endmodule``. The result is behaviourally
    identical to the original on its original ports; the new ports let a miter
    assert the state-matching invariant an unbounded k-induction proof needs (see
    the module docstring for why hierarchical references and ``bind`` do not work
    with yosys 0.33).

    Parameters
    ----------
    verilog : str
        Verilog source containing ``top``.
    top : str
        Module to instrument.
    taps : list[StateTap]
        The taps to add. Must be non-empty; port names must be unique and must
        not collide with an existing port.

    Returns
    -------
    str
        The instrumented Verilog source.

    Raises
    ------
    ValueError
        If ``taps`` is empty, a tap port is duplicated or already declared, or the
        module / its ``endmodule`` cannot be located.
    """
    if not taps:
        raise ValueError("taps must not be empty")

    tap_names = [t.port for t in taps]
    if len(set(tap_names)) != len(tap_names):
        raise ValueError("tap port names must be unique")

    existing = {m.group("name") for m in _PORT_RE.finditer(_module_header(verilog, top))}
    for name in tap_names:
        if name in existing:
            raise ValueError(f"tap port {name!r} already exists in module {top!r}")

    open_idx, close_idx = _port_list_bounds(verilog, top)
    decls = ",\n    " + ",\n    ".join(t.declaration() for t in taps)
    header_instrumented = verilog[:close_idx] + decls + "\n" + verilog[close_idx:]

    body_start = close_idx + len(decls) + 1
    end_match = re.compile(r"\bendmodule\b").search(header_instrumented, body_start)
    if end_match is None:
        raise ValueError(f"endmodule for module {top!r} not found")
    end_idx = end_match.start()
    assigns = "\n".join(t.assignment() for t in taps) + "\n"
    return header_instrumented[:end_idx] + assigns + header_instrumented[end_idx:]
