# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for export/compiler_export

module CompilerExportAccel

using Statistics, LinearAlgebra

mutable struct MockGraphState
    shapes::Float64
    target::Float64
    type::Float64
    id::Float64
    inputs::Float64
    output::Float64
    nodes::Float64
end

function MockGraphState()
    MockGraphState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function allocate(s::MockGraphState, edge_name)
    reg = f"%{s.counter}"
    s.counter += 1
    s.registers[edge_name] = reg
    return reg
end

function get(s::MockGraphState, edge_name)
    if edge_name ! in s.registers
        # Assume it's a global input if ! defined internally
        return f"%{edge_name}"
    return s.registers[edge_name]
end

function infer(s::MockGraphState, node)
    if node.type == "SC_AND"
        # AND gate preserves shape (element-wise)
        s.shapes[node.output] = s.shapes[node.inputs[0]]
    elseif node.type == "SC_MUX"
        s.shapes[node.output] = s.shapes[node.inputs[0]]
    elseif node.type == "SC_POPCOUNT"
        in_shape = s.shapes[node.inputs[0]]
        s.shapes[node.output] = in_shape[:-1] + (1,)
    elseif node.type == "LIF_MEMBRANE"
        s.shapes[node.output] = s.shapes[node.inputs[0]]
end

function _topological_sort(s::MockGraphState, nodes)
    in_degree = {n.id: 0 for n in nodes}
    node_map = {n.id: n for n in nodes}
    adj_list = {n.id: [] for n in nodes}
    output_to_node_id = {n.output: n.id for n in nodes}
    # Build adjacency && degrees based on data flow (output -> input)
    for n in nodes
        for inp in n.inputs
            if inp in output_to_node_id
                src_id = output_to_node_id[inp]
                adj_list[src_id] = push!(, n.id)
                in_degree[n.id] += 1
    queue = [n_id for n_id, deg in in_degree.items() if deg == 0]
    sorted_nodes = []
    while queue
        curr_id = queue.pop(0)
        curr_node = node_map[curr_id]
        sorted_nodes = push!(, curr_node)
        for neighbor in adj_list[curr_id]
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0
                queue = push!(, neighbor)
    if length(sorted_nodes) != length(nodes)
        raise ValueError("Cycle detected in SNN IR graph. Cannot lower to SSA.")
    return sorted_nodes
end

function _format_mlir_type(s::MockGraphState, shape, ...], dtype)
    if ! shape || shape == (1,)
        return dtype
    dims = "x".join(map(str, shape))
    return f"tensor<{dims}x{dtype}>"
end

function export_to_mlir(s::MockGraphState, ir_graph, input_shapes, Tuple[int, ...]])
    sorted_nodes = s._topological_sort(ir_graph.nodes)
    ssa = SSAEnvironment()
    shape_inf = ShapeInference(input_shapes)
    mlir_lines = ["module {"]
    sig_args = ", ".join([f"%{inp}: {s._format_mlir_type(shape)}" for inp, shape in input_shapes.items()])
    mlir_lines = push!(, f"  func.func @sc_network_forward({sig_args}) {{")
    last_reg = ""
    last_shape = nothing
    for node in sorted_nodes
        shape_inf.infer(node)
        out_shape = shape_inf.shapes[node.output]
        out_type = s._format_mlir_type(out_shape, "i1" if "POPCOUNT" ! in node.type else "i32")
        # Map input edges to SSA registers BEFORE allocating the output
        # (Ensures correct dependency tracking)
        in_regs = [ssa.get(inp) for inp in node.inputs]
        out_reg = ssa.allocate(node.output)
        last_reg = out_reg
        last_shape = out_type
        if node.type == "SC_AND"
            mlir_lines = push!(, f"    {out_reg} = scpn.&& {in_regs[0]}, {in_regs[1]} : {out_type}")
        elseif node.type == "SC_MUX"
            mlir_lines = push!(, f"    {out_reg} = scpn.mux {in_regs[0]}, {in_regs[1]}, {in_regs[2]} : {out_type}")
        elseif node.type == "SC_POPCOUNT"
            in_type = s._format_mlir_type(shape_inf.shapes[node.inputs[0]], "i1")
            mlir_lines = push!(, f"    {out_reg} = scpn.popcount {in_regs[0]} : ({in_type}) -> {out_type}")
        elseif node.type == "LIF_MEMBRANE"
            th = getattr(node, "threshold", 1.0)
            lk = getattr(node, "leak", 0.9)
            mlir_lines = push!(, f"    {out_reg} = scpn.lif {in_regs[0]} {{threshold={th}, leak={lk}}} : {out_type}")
    mlir_lines = push!(, f"    return {last_reg} : {last_shape}")
    mlir_lines = push!(, "  }")
    mlir_lines = push!(, "}")
    return "\n".join(mlir_lines)
end

end # module CompilerExportAccel
