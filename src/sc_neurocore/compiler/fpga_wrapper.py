# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA compilation wrapper

"""High-level wrapper for ODE to FPGA RTL compilation."""

from __future__ import annotations

from typing import Any

from ..neurons.equation_builder import EquationNeuron
from .verilog_compiler import compile_to_verilog


def equation_to_fpga(
    *equation_strings: str,
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, Any] | None = None,
    init: dict[str, Any] | None = None,
    constants: dict[str, Any] | None = None,
    dt: Any = 0.1,
    module_name: str = "sc_equation_neuron",
    data_width: int = 16,
    fraction: int = 8,
    units: str = "none",
    input_unit: Any | None = None,
) -> tuple[EquationNeuron, str]:
    """One-liner: ODE string → (Python neuron, Verilog RTL)."""
    from ..neurons.equation_builder import from_equations

    expanded: list[str] = []
    for s in equation_strings:
        expanded.extend(part.strip() for part in s.split(";") if part.strip())

    neuron = from_equations(
        *expanded,
        threshold=threshold,
        reset=reset,
        params=params,
        init=init,
        constants=constants,
        dt=dt,
        units=units,
        input_unit=input_unit,
    )
    verilog = compile_to_verilog(
        neuron,
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )
    return neuron, verilog
