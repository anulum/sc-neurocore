# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler Inspector backend for Studio

"""Verilog compilation and co-simulation adapters for the Studio equation playground."""

from __future__ import annotations

from typing import Any

from sc_neurocore.studio.compile_traceability import build_compile_traceability


def emit_sv_from_equation(
    equations: list[str],
    params: dict[str, float] | None = None,
    threshold: str | None = None,
    reset: str | None = None,
    init: dict[str, float] | None = None,
    module_name: str = "sc_ode_neuron",
) -> dict[str, Any]:
    """Direct equation → SystemVerilog via the Python equation compiler."""
    from sc_neurocore.compiler.equation_compiler import equation_to_fpga

    ir_repr, verilog = equation_to_fpga(
        *equations,
        threshold=threshold,
        reset=reset,
        params=params,
        init=init,
        module_name=module_name,
    )
    return {
        "verilog": verilog,
        "ir_repr": str(ir_repr),
        "chars": len(verilog),
        "module_name": module_name,
        "compile_traceability": build_compile_traceability(
            equations=equations,
            threshold=threshold,
            reset=reset,
            params=params,
            init=init,
            module_name=module_name,
            verilog=verilog,
        ).to_public_dict(),
    }


def cosim_traces(
    equations: list[str],
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, float] | None = None,
    init: dict[str, float] | None = None,
    dt: float = 0.1,
    duration: float = 100.0,
    current: float = 10.0,
    *,
    protocol: str = "constant",
    frequency_hz: float = 10.0,
    q_format: str = "Q8.8",
    overflow: str = "saturate",
    rounding: str = "truncate",
) -> dict[str, Any]:
    """Run the float64 reference and the bit-true fixed-point kernel side by side.

    This is the precision comparison of
    :func:`sc_neurocore.studio.precision_compare.precision_compare`: the
    fixed-point trace is the generated bit-true kernel executed natively,
    not a float run with rounded parameters.
    """
    from sc_neurocore.studio.precision_compare import precision_compare

    return precision_compare(
        equations=equations,
        threshold=threshold,
        reset=reset,
        params=params,
        init=init,
        dt=dt,
        duration=duration,
        current=current,
        protocol=protocol,
        frequency_hz=frequency_hz,
        q_format=q_format,
        overflow=overflow,
        rounding=rounding,
    )
