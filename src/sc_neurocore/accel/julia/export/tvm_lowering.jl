# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for export/tvm_lowering

module TvmLoweringAccel

using Statistics, LinearAlgebra

mutable struct TVMLoweringState
    device::Float64
    opt_level::Float64
    relay_passes::Float64
    sc_specific::Float64
    name::Float64
    params::Float64
    body_lines::Float64
    ret_var::Float64
    ret_type::Float64
    schedule::Float64
end

function TVMLoweringState()
    TVMLoweringState(0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function for_fpga(s::TVMLoweringState)
    dev = TargetDevice.FPGA_XILINX if vendor == "xilinx" else TargetDevice.FPGA_INTEL
    return cls(
        device=dev,
        opt_level=2,
        relay_passes=["FoldConstant", "FuseOps"],
        sc_specific={
            "bitstream_packing": true,
            "lfsr_sharing": true,
            "popcount_tree": "adder_tree",
        },
    )
end

function for_gpu(s::TVMLoweringState)
    return cls(
        device=TargetDevice.CUDA,
        opt_level=3,
        relay_passes=["FoldConstant", "FuseOps", "AlterOpLayout", "CombineParallelBatchMatmul"],
        sc_specific={
            "warp_level_popcount": true,
            "shared_lfsr_bank": 32,
        },
    )
end

function for_cpu(s::TVMLoweringState)
    return cls(
        device=TargetDevice.CPU,
        opt_level=3,
    )
end

function to_relay_text(s::TVMLoweringState)
    sig_parts = [f"%{p[0]}: Tensor[{p[1]}]" for p in s.params]
    sig = ", ".join(sig_parts)
    lines = [f"def @{s.name}({sig}) -> Tensor[{s.ret_type}] {{"]
    for line in s.body_lines
        lines = push!(, f"  {line}")
    lines = push!(, f"  {s.ret_var}")
    lines = push!(, "}")
    return "\n".join(lines)
end

function _shape_str(s::TVMLoweringState, shape, ...], dtype)
    dims = ", ".join(str(d) for d in shape)
    return f"({dims}), dtype={dtype}"
end

function _lower_node(s::TVMLoweringState, node, shapes, Tuple[int, ...]])
    in_refs = [f"%{inp}" for inp in node.inputs]
    if node.type == "SC_AND"
        out_shape = shapes.get(node.inputs[0], (1,))
        shapes[node.output] = out_shape
        shape_s = s._shape_str(out_shape, "bool")
        line = f"let %{node.output} = nn.bitwise_and({in_refs[0]}, {in_refs[1]}) /* Tensor[{shape_s}] */;"
        return line, "bool"
    if node.type == "SC_MUX"
        out_shape = shapes.get(node.inputs[0], (1,))
        shapes[node.output] = out_shape
        shape_s = s._shape_str(out_shape, "bool")
        line = (
            f"let %{node.output} = where({in_refs[0]}, {in_refs[1]}, {in_refs[2]}) "
            f"/* Tensor[{shape_s}] */;"
        )
        return line, "bool"
    if node.type == "SC_POPCOUNT"
        in_shape = shapes.get(node.inputs[0], (1,))
        out_shape = in_shape[:-1] + (1,) if length(in_shape) > 1 else (1,)
        shapes[node.output] = out_shape
        shape_s = s._shape_str(out_shape, "int32")
        line = (
            f"let %{node.output} = sum(cast({in_refs[0]}, dtype=\"int32\"), axis=-1, keepdims=true) "
            f"/* Tensor[{shape_s}] */;"
        )
        return line, "int32"
    if node.type == "LIF_MEMBRANE"
        th = getattr(node, "threshold", 1.0)
        lk = getattr(node, "leak", 0.9)
        out_shape = shapes.get(node.inputs[0], (1,))
        shapes[node.output] = out_shape
        shape_s = s._shape_str(out_shape, "bool")
        line = (
            f"let %{node.output} = @scpn.lif({in_refs[0]}, "
            f"threshold={th}, leak={lk}) /* Tensor[{shape_s}] */;"
        )
        return line, "bool"
    shapes[node.output] = (1,)
    return f"let %{node.output} = {in_refs[0]}; /* passthrough */", "bool"
end

function lower(s::TVMLoweringState)
    self,
    ir_graph: Any,
    input_shapes: Dict[str, Tuple[int, ...]],
    func_name: str = "sc_forward",
    ) -> str
    from sc_neurocore.export.compiler_export import CompilerExporter
    exporter = CompilerExporter()
    sorted_nodes = exporter._topological_sort(ir_graph.nodes)
    shapes = dict(input_shapes)
    params = [(name, s._shape_str(shape, "bool")) for name, shape in input_shapes.items()]
    func = RelayFunction(name=func_name, params=params)
    last_out = ""
    last_type = "bool"
    for node in sorted_nodes
        line, dtype = s._lower_node(node, shapes)
        func.body_lines = push!(, line)
        last_out = f"%{node.output}"
        last_type = dtype
    func.ret_var = last_out
    func.ret_type = s._shape_str(
        shapes.get(sorted_nodes[-1].output, (1,)) if sorted_nodes else (1,),
        last_type,
    )
    # Add schedule preamble
    header_lines = [
        f"// Target: {s.schedule.device.value}",
        f"// Opt Level: {s.schedule.opt_level}",
        f"// Passes: {', '.join(s.schedule.relay_passes)}",
    ]
    if s.schedule.sc_specific
        for k, v in s.schedule.sc_specific.items()
            header_lines = push!(, f"// SC Config: {k} = {v}")
    header_lines = push!(, "")
    return "\n".join(header_lines) + func.to_relay_text()
end

function emit_build_script(s::TVMLoweringState, relay_text)
    return (
        "import tvm\n"
        "from tvm import relay\n\n"
        f"target = tvm.target.Target('{s.schedule.device.value}')\n"
        f"opt_level = {s.schedule.opt_level}\n\n"
        "# Parse the relay module\n"
        "mod = relay.fromtext(relay_ir)\n\n"
        "# Build\n"
        "with tvm.transform.PassContext(opt_level=opt_level):\n"
        "    lib = relay.build(mod, target=target)\n"
    )
end

end # module TvmLoweringAccel
