# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reversible logic synthesizer

"""Synthesize reversible (lossless) logic for zero-energy computation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ReversibleNetlist:
    """Reversible logic netlist metrics.

    Attributes
    ----------
    toffoli_gates : int
    fredkin_gates : int
    ancilla_bits : int
    landauer_dissipation_kt : float
    """

    toffoli_gates: int
    fredkin_gates: int
    ancilla_bits: int
    landauer_dissipation_kt: float


def synthesize_reversible_logic(equations: dict[str, str], bits: int = 16) -> ReversibleNetlist:
    """Synthesize reversible (lossless) logic for zero-energy computation."""
    toffoli = 0
    fredkin = 0
    ancilla = 0

    for expr in equations.values():
        ops_add = expr.count("+") + expr.count("-")
        ops_mul = expr.count("*") + expr.count("/")

        toffoli += ops_add * (3 * bits)
        ancilla += ops_add * bits

        toffoli += ops_mul * (bits * bits)
        fredkin += ops_mul * (bits * bits)
        ancilla += ops_mul * (bits * bits)

    dissipation = ancilla * math.log(2)

    return ReversibleNetlist(
        toffoli_gates=toffoli,
        fredkin_gates=fredkin,
        ancilla_bits=ancilla,
        landauer_dissipation_kt=round(dissipation, 2),
    )
