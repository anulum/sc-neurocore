# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ONNX Export

"""Zero-dependency ONNX exporter for SC-NeuroCore IR graphs.

Maps SC-IR nodes to ONNX-compatible graph representation with custom
operator set `sc.neurocore`. No ONNX runtime or protobuf dependency
required — emits a self-contained dict-based graph that can be
serialized to JSON or consumed by downstream tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

ONNX_OPSET_VERSION = 18
SCPN_DOMAIN = "sc.neurocore"
SCPN_OPSET_VERSION = 1


@dataclass
class ONNXTensorType:
    """Tensor element type and static shape for the JSON ONNX model.

    Parameters
    ----------
    elem_type:
        ONNX tensor element type id.
    shape:
        Static tensor dimensions.
    """

    elem_type: int  # 1=float, 2=uint8, 3=int8, 6=int32, 7=int64, 9=bool
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the ONNX tensor-type dictionary representation."""
        return {
            "elem_type": self.elem_type,
            "shape": {"dim": [{"dim_value": d} for d in self.shape]},
        }


@dataclass
class ONNXNode:
    """Custom-domain ONNX node for a lowered stochastic-computing operation.

    Parameters
    ----------
    op_type:
        ONNX operator type.
    domain:
        Operator domain.
    inputs:
        Input tensor names.
    outputs:
        Output tensor names.
    name:
        Stable node name.
    attributes:
        Optional scalar operator attributes.
    """

    op_type: str
    domain: str
    inputs: list[str]
    outputs: list[str]
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the ONNX node dictionary representation."""
        d: dict[str, Any] = {
            "op_type": self.op_type,
            "domain": self.domain,
            "input": self.inputs,
            "output": self.outputs,
            "name": self.name,
        }
        if self.attributes:
            d["attribute"] = [
                {"name": k, "type": "FLOAT" if isinstance(v, float) else "INT", "value": v}
                for k, v in self.attributes.items()
            ]
        return d


@dataclass
class ONNXGraph:
    """JSON-serializable ONNX model envelope.

    Parameters
    ----------
    name:
        ONNX graph name.
    nodes:
        Lowered ONNX nodes.
    inputs:
        Named graph inputs and tensor types.
    outputs:
        Named graph outputs and tensor types.
    metadata:
        String metadata entries attached to the model.
    """

    name: str
    nodes: list[ONNXNode] = field(default_factory=list)
    inputs: list[tuple[str, ONNXTensorType]] = field(default_factory=list)
    outputs: list[tuple[str, ONNXTensorType]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete ONNX model dictionary representation."""
        return {
            "ir_version": 9,
            "opset_import": [
                {"domain": "", "version": ONNX_OPSET_VERSION},
                {"domain": SCPN_DOMAIN, "version": SCPN_OPSET_VERSION},
            ],
            "graph": {
                "name": self.name,
                "node": [n.to_dict() for n in self.nodes],
                "input": [
                    {"name": name, "type": {"tensor_type": tt.to_dict()}}
                    for name, tt in self.inputs
                ],
                "output": [
                    {"name": name, "type": {"tensor_type": tt.to_dict()}}
                    for name, tt in self.outputs
                ],
            },
            "metadata_props": [{"key": k, "value": v} for k, v in self.metadata.items()],
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the complete ONNX model as formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# SC-IR to ONNX mapping
SC_OP_MAP = {
    "SC_AND": "ScAnd",
    "SC_MUX": "ScMux",
    "SC_POPCOUNT": "ScPopcount",
    "LIF_MEMBRANE": "LifNeuron",
}


class ONNXExporter:
    """Export SC-NeuroCore IR graphs to ONNX-compatible dictionaries.

    Parameters
    ----------
    graph_name:
        Name assigned to the emitted ONNX graph.
    """

    def __init__(self, graph_name: str = "sc_network") -> None:
        self.graph_name = graph_name

    def _infer_type(self, node_type: str, shape: tuple[int, ...]) -> ONNXTensorType:
        if node_type == "SC_POPCOUNT":
            return ONNXTensorType(elem_type=6, shape=shape)  # int32
        return ONNXTensorType(elem_type=9, shape=shape)  # bool for SC bitstreams

    def _infer_shape(
        self,
        node_type: str,
        inputs: list[str],
        shapes: dict[str, tuple[int, ...]],
    ) -> tuple[int, ...]:
        if node_type in ("SC_AND", "SC_MUX", "LIF_MEMBRANE"):
            return shapes.get(inputs[0], (1,))
        if node_type == "SC_POPCOUNT":
            in_shape = shapes.get(inputs[0], (1,))
            return in_shape[:-1] + (1,) if len(in_shape) > 1 else (1,)
        raise ValueError(f"No ONNX shape rule for mapped SC-IR node type {node_type!r}")

    def export(
        self,
        ir_graph: Any,
        input_shapes: dict[str, tuple[int, ...]],
        metadata: dict[str, str] | None = None,
    ) -> ONNXGraph:
        """Convert an SC-IR graph to an ONNX graph representation.

        Parameters
        ----------
        ir_graph:
            SC-IR graph-like object with a ``nodes`` sequence.
        input_shapes:
            Mapping from input tensor names to static dimensions.
        metadata:
            Optional string metadata to attach to the emitted graph.

        Returns
        -------
        ONNXGraph
            JSON-serializable ONNX graph envelope.

        Raises
        ------
        ValueError
            If a mapped SC-IR operator has no shape inference rule.
        """
        from sc_neurocore.export.compiler_export import CompilerExporter

        exporter = CompilerExporter()
        sorted_nodes = exporter._topological_sort(ir_graph.nodes)

        graph = ONNXGraph(name=self.graph_name, metadata=metadata or {})

        # Register inputs
        for inp_name, shape in input_shapes.items():
            graph.inputs.append((inp_name, ONNXTensorType(elem_type=9, shape=shape)))

        # Track shapes for inference
        shapes: dict[str, tuple[int, ...]] = dict(input_shapes)

        # Convert nodes
        last_output = ""
        last_node_type = ""
        for node in sorted_nodes:
            op = SC_OP_MAP.get(node.type)
            if op is None:
                continue

            out_shape = self._infer_shape(node.type, list(node.inputs), shapes)
            shapes[node.output] = out_shape

            # Build ONNX node
            attrs: dict[str, Any] = {}
            if node.type == "LIF_MEMBRANE":
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
            graph.nodes.append(onnx_node)
            last_output = node.output
            last_node_type = node.type

        # Register final output
        if last_output and last_output in shapes:
            graph.outputs.append(
                (last_output, self._infer_type(last_node_type, shapes[last_output]))
            )

        return graph
