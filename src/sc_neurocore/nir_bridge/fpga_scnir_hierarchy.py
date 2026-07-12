# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler
"""SC-NIR hierarchy boundary and semantic weight-stream RTL emission."""

import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..hdl_gen._ident import sanitize_ident
from ..ir.scnir_schema import (
    SCNIRDocument,
    SCNIRHierarchyInstance,
    SCNIRHierarchyPort,
)
from .fpga_connection_routing import _signed_hex
from .quantise_params import QuantisedGraph

_SCNIR_STREAM_FRAGMENT_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def _scnir_stream_fragment(value: str) -> str:
    cleaned = _SCNIR_STREAM_FRAGMENT_RE.sub("_", value.strip())
    cleaned = cleaned.strip("_.:-")
    if not cleaned:
        cleaned = "stream"
    if not cleaned[0].isalpha():
        cleaned = f"s_{cleaned}"
    return cleaned[:96]


def _scnir_connection_stream_id(src: str, dst: str) -> str:
    return f"conn.{_scnir_stream_fragment(src)}_to_{_scnir_stream_fragment(dst)}.weight"


def resolve_hierarchy_weight_literals(
    document: SCNIRDocument,
    qgraph: QuantisedGraph,
) -> dict[str, tuple[int, ...]]:
    """Resolve flattened weights referenced by hierarchy output ports.

    Parameters
    ----------
    document : SCNIRDocument
        Typed hierarchy and stream metadata for the compilation.
    qgraph : QuantisedGraph
        Quantised graph that owns the referenced connection weights.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Flattened integer weights keyed by semantic SC-NIR stream identifier.

    Raises
    ------
    ValueError
        If a weight port references an unknown stream or incompatible packed width.
    """
    weights_by_stream: dict[str, np.ndarray[Any, Any]] = {
        _scnir_connection_stream_id(str(conn.src), str(conn.dst)): np.asarray(
            conn.weights,
            dtype=np.int64,
        )
        for conn in qgraph.connections
    }
    literals: dict[str, tuple[int, ...]] = {}
    for instance in document.hierarchy:
        for port in instance.ports:
            if port.direction != "output" or port.signal_kind != "weight":
                continue
            weights = weights_by_stream.get(port.stream_id)
            if weights is None:
                raise ValueError(
                    f"SC-NIR hierarchy port {port.port_name!r} references unknown "
                    f"weight stream {port.stream_id!r}"
                )
            flat = weights.reshape(-1)
            if port.bit_width % int(flat.size) != 0:
                raise ValueError(
                    f"SC-NIR hierarchy weight port {port.port_name!r} bit width "
                    f"{port.bit_width} is not divisible by flattened weight count {flat.size}"
                )
            literals[port.stream_id] = tuple(int(value) for value in flat)
    return literals


def _hierarchy_output_wires_by_stream(
    hierarchy: Sequence[SCNIRHierarchyInstance],
    *,
    semantic_stream_ids: set[str],
) -> dict[str, tuple[str, int]]:
    """Return top-level hierarchy output wire names keyed by SC-NIR stream id."""
    wires: dict[str, tuple[str, int]] = {}
    for instance in hierarchy:
        module_name = sanitize_ident(instance.module_name, context="hierarchy module name")
        for port in instance.ports:
            if port.direction != "output" or port.stream_id not in semantic_stream_ids:
                continue
            port_name = sanitize_ident(port.port_name, context="hierarchy port name")
            wire_name = sanitize_ident(
                f"{module_name}__{port_name}",
                context="hierarchy top-level wire name",
            )
            if port.stream_id in wires:
                raise ValueError(f"duplicate hierarchy output for stream {port.stream_id!r}")
            wires[port.stream_id] = (wire_name, port.bit_width)
    return wires


def build_scnir_hierarchy_modules(
    document: SCNIRDocument,
    *,
    weight_literals: dict[str, tuple[int, ...]],
) -> dict[str, str]:
    """Emit standalone boundary modules for preserved SC-NIR hierarchy instances.

    Parameters
    ----------
    document : SCNIRDocument
        Typed hierarchy metadata to lower.
    weight_literals : dict[str, tuple[int, ...]]
        Flattened fixed-point weights keyed by semantic stream identifier.

    Returns
    -------
    dict[str, str]
        Synthesisable Verilog keyed by hierarchy module name.

    Raises
    ------
    ValueError
        If module names collide or a hierarchy boundary is malformed.
    """
    modules: dict[str, str] = {}
    for instance in document.hierarchy:
        module_name = sanitize_ident(instance.module_name, context="hierarchy module name")
        if module_name in modules:
            raise ValueError(f"duplicate SC-NIR hierarchy module name {module_name!r}")
        modules[module_name] = _build_scnir_hierarchy_module(
            instance,
            module_name=module_name,
            weight_literals=weight_literals,
        )
    return modules


def _build_scnir_hierarchy_module(
    instance: SCNIRHierarchyInstance,
    *,
    module_name: str,
    weight_literals: dict[str, tuple[int, ...]],
) -> str:
    """Emit one synthesisable hierarchy boundary module from typed SC-NIR ports."""
    if not instance.ports:
        raise ValueError(f"SC-NIR hierarchy instance {instance.instance_id!r} has no ports")
    port_lines = [f"    {_hierarchy_port_declaration(port)}" for port in instance.ports]
    lines = [
        f"// Auto-generated SC-NIR hierarchy boundary: {module_name}",
        "// Scalar weight outputs may feed the generated top-level MAC path.",
        "`timescale 1ns / 1ps",
        "",
        f"module {module_name} (",
        ",\n".join(port_lines),
        ");",
        "",
    ]
    for port in instance.ports:
        lines.append(f"    // stream_id: {port.stream_id}")
        lines.append(f"    // signal_kind: {port.signal_kind}")
        if port.direction == "output":
            port_name = sanitize_ident(port.port_name, context="hierarchy port name")
            literals = weight_literals.get(port.stream_id)
            if literals is None:
                lines.append(f"    assign {port_name} = {_hierarchy_zero_literal(port)};")
            elif len(literals) == 1:
                lines.append(
                    f"    assign {port_name} = {_signed_hex(literals[0], port.bit_width)};"
                )
            else:
                if port.bit_width % len(literals) != 0:
                    raise ValueError(
                        f"SC-NIR hierarchy port {port.port_name!r} bit width "
                        f"{port.bit_width} is not divisible by literal count {len(literals)}"
                    )
                element_width = port.bit_width // len(literals)
                for index, literal in enumerate(literals):
                    offset = index * element_width
                    lines.append(
                        f"    assign {port_name}[{offset} +: {element_width}] = "
                        f"{_signed_hex(literal, element_width)};"
                    )
        lines.append("")
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def _hierarchy_port_declaration(port: SCNIRHierarchyPort) -> str:
    direction = port.direction
    port_name = sanitize_ident(port.port_name, context="hierarchy port name")
    if port.bit_width <= 0:
        raise ValueError(f"SC-NIR hierarchy port {port.port_name!r} has non-positive bit width")
    if port.bit_width == 1:
        return f"{direction} wire {port_name}"
    return f"{direction} wire signed [{port.bit_width - 1}:0] {port_name}"


def _hierarchy_zero_literal(port: SCNIRHierarchyPort) -> str:
    if port.bit_width == 1:
        return "1'b0"
    return f"{port.bit_width}'sd0"


def build_scnir_hierarchy_instance_block(
    hierarchy: Sequence[SCNIRHierarchyInstance],
    *,
    data_width: int,
) -> list[str]:
    """Emit top-level hierarchy instances and typed connecting wires.

    Parameters
    ----------
    hierarchy : Sequence[SCNIRHierarchyInstance]
        Typed hierarchy instances to connect at the network top.
    data_width : int
        Active fixed-point data width used for canonical weight wires.

    Returns
    -------
    list[str]
        Verilog declarations, zero-driven inputs, and instance blocks.

    Raises
    ------
    ValueError
        If a hierarchy port has a non-positive width.
    """
    if not hierarchy:
        return []
    lines = ["    // SC-NIR hierarchy contract instances"]
    for instance in hierarchy:
        module_name = sanitize_ident(instance.module_name, context="hierarchy module name")
        instance_name = sanitize_ident(
            f"{module_name}_hierarchy_inst",
            context="hierarchy instance name",
        )
        port_bindings: list[str] = []
        for port in instance.ports:
            port_name = sanitize_ident(port.port_name, context="hierarchy port name")
            wire_name = sanitize_ident(
                f"{module_name}__{port_name}",
                context="hierarchy top-level wire name",
            )
            lines.append(_hierarchy_top_wire_declaration(wire_name, port, data_width=data_width))
            if port.direction == "input":
                lines.append(f"    assign {wire_name} = {_hierarchy_zero_literal(port)};")
            port_bindings.append(f"        .{port_name}({wire_name})")
        lines.append(f"    {module_name} {instance_name} (")
        lines.append(",\n".join(port_bindings))
        lines.append("    );")
        lines.append("")
    return lines


def _hierarchy_top_wire_declaration(
    wire_name: str,
    port: SCNIRHierarchyPort,
    *,
    data_width: int,
) -> str:
    if port.bit_width <= 0:
        raise ValueError(f"SC-NIR hierarchy port {port.port_name!r} has non-positive bit width")
    if port.bit_width == 1:
        return f"    wire {wire_name};"
    if port.bit_width == data_width:
        return f"    wire signed [DATA_WIDTH - 1:0] {wire_name};"
    return f"    wire signed [{port.bit_width - 1}:0] {wire_name};"


def _hierarchy_weight_expr(
    hierarchy_output_wires: Mapping[str, tuple[str, int]],
    stream_id: str,
    *,
    weight: int,
    data_width: int,
    weight_index: int,
) -> str:
    wire = hierarchy_output_wires.get(stream_id)
    if wire is None:
        return _signed_hex(weight, data_width)
    wire_name, bit_width = wire
    if bit_width == data_width:
        return wire_name
    offset = weight_index * data_width
    if offset + data_width > bit_width:
        raise ValueError(
            f"SC-NIR hierarchy stream {stream_id!r} width {bit_width} cannot provide "
            f"weight index {weight_index}"
        )
    return f"{wire_name}[{offset} +: DATA_WIDTH]"
