# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/compiler

module CompilerAccel

using Statistics, LinearAlgebra

function build_ir_from_equation(equations, params, threshold, reset, dt)
    equations: list[str],
    params: dict[str, float] | nothing = nothing,
    threshold: str | nothing = nothing,
    reset: str | nothing = nothing,
    dt: float = 0.1,
    ) -> dict
    from sc_neurocore_engine import ScGraphBuilder, ir_print, ir_verify
    params = params || {}
    builder = ScGraphBuilder("ode_neuron")
    # Input ports
    clk = builder.input("clk", "bool")
    i_ext = builder.input("i_ext", "fixed<16,8>")
    # Encode parameters as fixed-point constants
    q88_params = {}
    for name, value in params.items()
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
    errors = ir_verify(graph)
    ir_text = ir_print(graph)
    return {
        "ir_text": ir_text,
        "errors": errors if errors else [],
        "n_ops": length(graph),
        "n_inputs": graph.num_inputs(),
        "n_outputs": graph.num_outputs(),
        "graph_name": graph.name,
        "params_q88": {k: int(round(v * 256)) for k, v in (params || {}).items()},
    }
end

function verify_ir(ir_text)
    from sc_neurocore_engine import ir_parse, ir_verify
    graph = ir_parse(ir_text)
    errors = ir_verify(graph)
    return {
        "valid": errors is nothing,
        "errors": errors if errors else [],
        "n_ops": length(graph),
        "graph_name": graph.name,
    }
end

function emit_systemverilog(ir_text)
    from sc_neurocore_engine import ir_parse, ir_emit_sv
    graph = ir_parse(ir_text)
    sv_source = ir_emit_sv(graph)
    return {
        "systemverilog": sv_source,
        "graph_name": graph.name,
        "chars": length(sv_source),
    }
end

function emit_sv_from_equation(equations, params, threshold, reset)
    equations: list[str],
    params: dict[str, float] | nothing = nothing,
    threshold: str | nothing = nothing,
    reset: str | nothing = nothing,
    ) -> dict
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
        "chars": length(verilog),
        "module_name": "sc_ode_neuron",
    }
end

function cosim_traces(equations, threshold, reset, params, init, dt, duration, current)
    equations: list[str],
    threshold: str | nothing = nothing,
    reset: str | nothing = nothing,
    params: dict[str, float] | nothing = nothing,
    init: dict[str, float] | nothing = nothing,
    dt: float = 0.1,
    duration: float = 100.0,
    current: float = 10.0,
    ) -> dict
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
end

end # module CompilerAccel
