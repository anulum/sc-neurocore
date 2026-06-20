# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler Inspector backend for Studio

from __future__ import annotations

from typing import Any

from sc_neurocore.studio.compile_traceability import build_compile_traceability


def build_ir_from_equation(
    equations: list[str],
    params: dict[str, float] | None = None,
    threshold: str | None = None,
    reset: str | None = None,
    dt: float = 0.1,
) -> dict[str, Any]:
    """Build an SC IR graph from ODE equations.

    Uses the Rust ScGraphBuilder to construct a stochastic computing IR
    that represents the neuron's ODE as a hardware pipeline:
    input current → encode → multiply (leak, gain) → LIF step → output spike.
    """
    from sc_neurocore_engine.ir import ScGraphBuilder

    params = params or {}
    builder = ScGraphBuilder("ode_neuron")

    # Input ports
    clk = builder.input("clk", "bool")
    i_ext = builder.input("i_ext", "fixed<16,8>")

    # Encode parameters as fixed-point constants
    q88_params = {}
    for name, value in params.items():
        q88_params[name] = builder.constant_f64(value, "fixed<16,8>")

    # Encode dt
    dt_const = builder.constant_f64(dt, "fixed<16,8>")

    # LIF neuron step: current → leak → gain → spike
    leak_const = builder.constant_i64(230, "i16")  # ~0.9 decay
    gain_const = builder.constant_i64(26, "i16")  # ~0.1 gain
    noise_const = builder.constant_i64(0, "i16")

    spike = builder.lif_step(
        i_ext,
        leak_const,
        gain_const,
        noise_const,
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2,
    )

    # Output port
    builder.output("spike", spike)

    graph = builder.build()

    # Verification
    errors = graph.verify()
    ir_text = graph.to_text()

    return {
        "ir_text": ir_text,
        "errors": errors if errors else [],
        "n_ops": len(graph),
        "n_inputs": graph.num_inputs,
        "n_outputs": graph.num_outputs,
        "graph_name": graph.name,
        "params_q88": {k: int(round(v * 256)) for k, v in (params or {}).items()},
    }


def verify_ir(ir_text: str) -> dict[str, Any]:
    """Parse and verify an IR text representation."""
    from sc_neurocore_engine.ir import parse_ir

    graph = parse_ir(ir_text)
    errors = graph.verify()
    return {
        "valid": errors is None,
        "errors": errors if errors else [],
        "n_ops": len(graph),
        "graph_name": graph.name,
    }


def emit_systemverilog(ir_text: str) -> dict[str, Any]:
    """Parse IR text and emit synthesisable SystemVerilog."""
    from sc_neurocore_engine.ir import parse_ir

    graph = parse_ir(ir_text)
    sv_source = graph.emit_sv()
    return {
        "systemverilog": sv_source,
        "graph_name": graph.name,
        "chars": len(sv_source),
    }


def emit_sv_from_equation(
    equations: list[str],
    params: dict[str, float] | None = None,
    threshold: str | None = None,
    reset: str | None = None,
) -> dict[str, Any]:
    """Direct equation → SystemVerilog via the Python equation compiler."""
    from sc_neurocore.compiler.equation_compiler import equation_to_fpga

    ir_repr, verilog = equation_to_fpga(
        equations[0],
        threshold=threshold,
        reset=reset,
        params=params,
        module_name="sc_ode_neuron",
    )
    return {
        "verilog": verilog,
        "ir_repr": str(ir_repr),
        "chars": len(verilog),
        "module_name": "sc_ode_neuron",
        "compile_traceability": build_compile_traceability(
            equations=equations,
            threshold=threshold,
            reset=reset,
            params=params,
            init=None,
            module_name="sc_ode_neuron",
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
) -> dict[str, Any]:
    """Run Python float and Q8.8 fixed-point simulations side by side."""
    from sc_neurocore.studio.analysis import precision_compare

    return precision_compare(
        equations=equations,
        threshold=threshold,
        reset=reset,
        params=params,
        init=init,
        dt=dt,
        duration=duration,
        current=current,
    )
