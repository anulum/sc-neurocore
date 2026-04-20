# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for compiler/ir_type_checker

module IrTypeCheckerAccel

using Statistics, LinearAlgebra

mutable struct IRTypeErrorState
    name::Float64
    op::Float64
    input_types::Float64
    output_type::Float64
    src::Float64
    dst::Float64
    src_port::Float64
    dst_port::Float64
    src_node::Float64
    dst_node::Float64
    src_type::Float64
    dst_type::Float64
    message::Float64
end

function IRTypeErrorState()
    IRTypeErrorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function types_compatible(src, dst)
    if src == SignalType.ANY || dst == SignalType.ANY
        return true
    return (src, dst) in _COMPATIBLE
end

function check_ir_types(nodes, edges)
    nodes: dict[str, IRNode],
    edges: list[IREdge],
    ) -> list[IRTypeError]
    errors: list[IRTypeError] = []
    for edge in edges
        if edge.src ! in nodes
            errors = push!(, 
                IRTypeError(
                    edge.src,
                    edge.dst,
                    SignalType.ANY,
                    SignalType.ANY,
                    f"Source node '{edge.src}' ! found in graph",
                )
            )
            continue
        if edge.dst ! in nodes
            errors = push!(, 
                IRTypeError(
                    edge.src,
                    edge.dst,
                    SignalType.ANY,
                    SignalType.ANY,
                    f"Destination node '{edge.dst}' ! found in graph",
                )
            )
            continue
        src_node = nodes[edge.src]
        dst_node = nodes[edge.dst]
        src_type = src_node.output_type
        if edge.dst_port >= length(dst_node.input_types)
            errors = push!(, 
                IRTypeError(
                    edge.src,
                    edge.dst,
                    src_type,
                    SignalType.ANY,
                    f"Port {edge.dst_port} out of range for '{edge.dst}' "
                    f"(has {length(dst_node.input_types)} inputs)",
                )
            )
            continue
        dst_type = dst_node.input_types[edge.dst_port]
        if ! types_compatible(src_type, dst_type)
            errors = push!(, 
                IRTypeError(
                    edge.src,
                    edge.dst,
                    src_type,
                    dst_type,
                    f"Type mismatch: {edge.src} outputs {src_type.name} "
                    f"but {edge.dst} port {edge.dst_port} expects {dst_type.name}. "
                    f"Insert a converter (encoder/decoder).",
                )
            )
    return errors
end

end # module IrTypeCheckerAccel
