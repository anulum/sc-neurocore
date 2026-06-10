# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Omni-paradigm dispatcher

"""Dispatch ODE variables across heterogeneous computing paradigms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OmniDispatchMap:
    """Mapping of variables to heterogeneous computing backends.

    Attributes
    ----------
    cmos_variables : list[str]
        Standard digital/SRAM variables.
    thermodynamic_variables : list[str]
        Stochastic/noise-driven variables (RRAM, PCM).
    optical_variables : list[str]
        High-bandwidth weight/sum variables (Photonic).
    quantum_variables : list[str]
        Entangled/superposition variables (Superconducting).
    """

    cmos_variables: list[str]
    thermodynamic_variables: list[str]
    optical_variables: list[str]
    quantum_variables: list[str]


def dispatch_omni_paradigm(equations: dict[str, str]) -> OmniDispatchMap:
    """Dispatch ODE variables across heterogeneous computing paradigms."""
    cmos, thermo, optic, quant = [], [], [], []

    for var, expr in equations.items():
        expr_lower = expr.lower()
        if "rand" in expr_lower or "noise" in expr_lower or "sigma" in expr_lower:
            thermo.append(var)
        elif "weight" in expr_lower or "sum" in expr_lower or "dot" in expr_lower:
            optic.append(var)
        elif "entangle" in expr_lower or "superpos" in expr_lower:
            quant.append(var)
        else:
            cmos.append(var)

    return OmniDispatchMap(cmos, thermo, optic, quant)
