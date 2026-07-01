# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for export/onnx_export

module OnnxExportAccel

using Statistics, LinearAlgebra

mutable struct ONNXExporterState
    elem_type::Float64
    shape::Float64
    op_type::Float64
    domain::Float64
    inputs::Float64
    outputs::Float64
    name::Float64
    attributes::Float64
    nodes::Float64
    metadata::Float64
    graph_name::Float64
end

function ONNXExporterState()
    ONNXExporterState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function to_dict(s::ONNXExporterState)
    return {
        "elem_type": s.elem_type,
        "shape": {"dim": [{"dim_value": d} for d in s.shape]},
    }
end

function to_dict(s::ONNXExporterState)
    d = {
        "op_type": s.op_type,
        "domain": s.domain,
        "input": s.inputs,
        "output": s.outputs,
        "name": s.name,
    }
    if s.attributes
        d["attribute"] = [
            {"name": k, "type": "FLOAT" if isinstance(v, float) else "INT", "value": v}
            for k, v in s.attributes.items()
        ]
    return d
end

function to_dict(s::ONNXExporterState)
    return {
        "ir_version": 9,
        "opset_import": [
            {"domain": "", "version": ONNX_OPSET_VERSION},
            {"domain": SCPN_DOMAIN, "version": SCPN_OPSET_VERSION},
        ],
        "graph": {
            "name": s.name,
            "node": [n.to_dict() for n in s.nodes],
            "input": [
                {"name": name, "type": {"tensor_type": tt.to_dict()}}
                for name, tt in s.inputs
            ],
            "output": [
                {"name": name, "type": {"tensor_type": tt.to_dict()}}
                for name, tt in s.outputs
            ],
        },
        "metadata_props": [
            {"key": k, "value": v} for k, v in s.metadata.items()
        ],
    }
end

function to_json(s::ONNXExporterState, indent)
    return json.dumps(s.to_dict(), indent=indent)
end

function _infer_type(s::ONNXExporterState, node_type, shape, ...])
    if node_type == "SC_POPCOUNT"
        return ONNXTensorType(elem_type=6, shape=shape)  # int32
    return ONNXTensorType(elem_type=9, shape=shape)
end

function export(s::ONNXExporterState)
    self,
    ir_graph: Any,
    input_shapes: Dict[str, Tuple[int, ...]],
    metadata: Dict[str, str] | nothing = nothing,
    ) -> ONNXGraph
    from sc_neurocore.export.compiler_export import CompilerExporter
    exporter = CompilerExporter()
    sorted_nodes = exporter._topological_sort(ir_graph.nodes)
    graph = ONNXGraph(name=s.graph_name, metadata=metadata || {})
    # Register inputs
    for inp_name, shape in input_shapes.items()
        graph.inputs = push!(,
            (inp_name, ONNXTensorType(elem_type=9, shape=shape))
        )
    # Track shapes for inference
    shapes: Dict[str, Tuple[int, ...]] = dict(input_shapes)
    # Convert nodes
    last_output = ""
    last_node_type = ""
    for node in sorted_nodes
        op = SC_OP_MAP.get(node.type)
        if op is nothing
            continue
        # Shape inference
        if node.type in ("SC_AND", "SC_MUX", "LIF_MEMBRANE")
            out_shape = shapes.get(node.inputs[0], (1,))
        elseif node.type == "SC_POPCOUNT"
            in_shape = shapes.get(node.inputs[0], (1,))
            out_shape = in_shape[:-1] + (1,) if length(in_shape) > 1 else (1,)
        else
            raise(ValueError("No ONNX shape rule for mapped SC-IR node type"))
        shapes[node.output] = out_shape
        # Build ONNX node
        attrs = {}
        if node.type == "LIF_MEMBRANE"
            attrs["threshold"] = getattr(node, "threshold", 1.0)
            attrs["leak"] = getattr(node, "leak", 0.9)
        onnx_node = ONNXNode(
            op_type=op,
            domain=SCPN_DOMAIN,
            inputs=list(node.inputs),
            outputs=[node.output],
            name=f"{op}_{node.id}",
            attributes=attrs,
        )
        graph.nodes = push!(, onnx_node)
        last_output = node.output
        last_node_type = node.type
    # Register final output
    if last_output && last_output in shapes
        graph.outputs = push!(,
            (last_output, s._infer_type(last_node_type, shapes[last_output]))
        )
    return graph
end

end # module OnnxExportAccel
