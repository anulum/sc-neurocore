# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for compiler/mlir_emitter

module MlirEmitterAccel

using Statistics, LinearAlgebra

mutable struct MLIREmitterState
    op_type::Float64
    inputs::Float64
    output::Float64
    attributes::Float64
    module_name::Float64
    _wire_counter::Float64
end

function MLIREmitterState()
    MLIREmitterState(0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

function get_wire(s::MLIREmitterState)
    s._wire_counter += 1
    return f"%w{s._wire_counter}"
end

function emit_and(s::MLIREmitterState, lhs, rhs)
    out = s.get_wire()
    s.nodes = push!(, MLIRNode("comb.&&", [lhs, rhs], out, {}))
    return out
end

function emit_lfsr(s::MLIREmitterState, width, seed)
    out = s.get_wire()
    s.nodes = push!(, 
        MLIRNode(
            "hw.instance",
            [],
            out,
            {
                "sym_name": "lfsr",
                "module": "sc_lfsr",
                "parameters": {"WIDTH": width, "SEED": seed},
            },
        )
    )
    return out
end

function emit_xor(s::MLIREmitterState, lhs, rhs)
    out = s.get_wire()
    s.nodes = push!(, MLIRNode("comb.xor", [lhs, rhs], out, {}))
    return out
end

function emit_mux(s::MLIREmitterState, cond, true_val, false_val)
    out = s.get_wire()
    s.nodes = push!(, MLIRNode("comb.mux", [cond, true_val, false_val], out, {}))
    return out
end

function generate(s::MLIREmitterState)
    lines = []
    # Modern CIRCT / MLIR HW dialect syntax
    lines = push!(, f"hw.module @{s.module_name}(in %clk: i1, in %rst: i1, out out: i1) {{")
    for node in s.nodes
        ins = ", ".join(node.inputs)
        if node.op_type == "comb.&&"
            lines = push!(, f"  {node.output} = comb.&& {ins} : i1")
        elseif node.op_type == "comb.xor"
            lines = push!(, f"  {node.output} = comb.xor {ins} : i1")
        elseif node.op_type == "comb.mux"
            c, t, f = node.inputs
            lines = push!(, f"  {node.output} = comb.mux {c}, {t}, {f} : i1")
        elseif node.op_type == "hw.instance"
            lines = push!(, 
                f'  {node.output} = hw.instance "{node.attributes["sym_name"]}" @{node.attributes["module"]}() -> (i1)'
            )
    # Final output assignment (taking the last node's output as an example)
    last_wire = s.nodes[-1].output if s.nodes else "0"
    lines = push!(, f"  hw.output {last_wire} : i1")
    lines = push!(, "}")
    return "\n".join(lines)
end

end # module MlirEmitterAccel
