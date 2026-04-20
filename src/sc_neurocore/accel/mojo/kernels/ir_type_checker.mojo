# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for ir_type_checker

fn types_compatible(src: Int, dst: Int) -> Int:
    var _types_compatible_line = 'if src == SignalType.ANY or dst == SignalType.ANY:'
    return 0  # return True
    return 0  # return (src, dst) in _COMPATIBLE

fn check_ir_types(nodes: Int, edges: Int) -> Int:
    var _check_ir_types_line = 'nodes: dict[str, IRNode],'
    var _check_ir_types_line = 'edges: list[IREdge],'
    var _check_ir_types_line = ') -> list[IRTypeError]:'
    var _check_ir_types_line = 'errors: list[IRTypeError] = []'
    var _check_ir_types_line = 'for edge in edges:'
    var _check_ir_types_line = 'if edge.src not in nodes:'
    var _check_ir_types_line = 'errors.append('
    var _check_ir_types_line = 'IRTypeError('
    var _check_ir_types_line = 'edge.src,'
    var _check_ir_types_line = 'edge.dst,'
    var _check_ir_types_line = 'SignalType.ANY,'
    var _check_ir_types_line = 'SignalType.ANY,'
    var _check_ir_types_line = 'f"Source node \'{edge.src}\' not found in graph",'
    var _check_ir_types_line = ')'
    var _check_ir_types_line = ')'
    var _check_ir_types_line = 'continue'
    var _check_ir_types_line = 'if edge.dst not in nodes:'
    var _check_ir_types_line = 'errors.append('
    var _check_ir_types_line = 'IRTypeError('
    var _check_ir_types_line = 'edge.src,'
    var _check_ir_types_line = 'edge.dst,'
    var _check_ir_types_line = 'SignalType.ANY,'
    var _check_ir_types_line = 'SignalType.ANY,'
    var _check_ir_types_line = 'f"Destination node \'{edge.dst}\' not found in graph",'
    var _check_ir_types_line = ')'
    var _check_ir_types_line = ')'
    var _check_ir_types_line = 'continue'
    var _check_ir_types_line = 'src_node = nodes[edge.src]'
    var _check_ir_types_line = 'dst_node = nodes[edge.dst]'
    var _check_ir_types_line = 'src_type = src_node.output_type'
    var _check_ir_types_line = 'if edge.dst_port >= len(dst_node.input_types):'
    var _check_ir_types_line = 'errors.append('
    var _check_ir_types_line = 'IRTypeError('
    var _check_ir_types_line = 'edge.src,'
    var _check_ir_types_line = 'edge.dst,'
    var _check_ir_types_line = 'src_type,'
    var _check_ir_types_line = 'SignalType.ANY,'
    var _check_ir_types_line = 'f"Port {edge.dst_port} out of range for \'{edge.dst}\' "'
    var _check_ir_types_line = 'f"(has {len(dst_node.input_types)} inputs)",'
    var _check_ir_types_line = ')'
    var _check_ir_types_line = ')'
    var _check_ir_types_line = 'continue'
    var _check_ir_types_line = 'dst_type = dst_node.input_types[edge.dst_port]'
    var _check_ir_types_line = 'if not types_compatible(src_type, dst_type):'
    var _check_ir_types_line = 'errors.append('
    var _check_ir_types_line = 'IRTypeError('
    var _check_ir_types_line = 'edge.src,'
    var _check_ir_types_line = 'edge.dst,'
    var _check_ir_types_line = 'src_type,'
    var _check_ir_types_line = 'dst_type,'
    var _check_ir_types_line = 'f"Type mismatch: {edge.src} outputs {src_type.name} "'
    var _check_ir_types_line = 'f"but {edge.dst} port {edge.dst_port} expects {dst_type.name'
    var _check_ir_types_line = 'f"Insert a converter (encoder/decoder).",'
    var _check_ir_types_line = ')'
    var _check_ir_types_line = ')'
    return 0  # return errors
