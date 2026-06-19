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
from typing import Any, Dict, List, Tuple

ONNX_OPSET_VERSION = 18
SCPN_DOMAIN = "sc.neurocore"
SCPN_OPSET_VERSION = 1


@dataclass
class ONNXTensorType:
    elem_type: int  # 1=float, 2=uint8, 3=int8, 6=int32, 7=int64, 9=bool
    shape: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elem_type": self.elem_type,
            "shape": {"dim": [{"dim_value": d} for d in self.shape]},
        }


@dataclass
class ONNXNode:
    op_type: str
    domain: str
    inputs: List[str]
    outputs: List[str]
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
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
    name: str
    nodes: List[ONNXNode] = field(default_factory=list)
    inputs: List[Tuple[str, ONNXTensorType]] = field(default_factory=list)
    outputs: List[Tuple[str, ONNXTensorType]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
        return json.dumps(self.to_dict(), indent=indent)


# SC-IR to ONNX mapping
SC_OP_MAP = {
    "SC_AND": "ScAnd",
    "SC_MUX": "ScMux",
    "SC_POPCOUNT": "ScPopcount",
    "LIF_MEMBRANE": "LifNeuron",
}


class ONNXExporter:
    """Exports SC-NeuroCore IR graphs to ONNX-compatible representation."""

    def __init__(self, graph_name: str = "sc_network"):
        self.graph_name = graph_name

    def _infer_type(self, node_type: str, shape: Tuple[int, ...]) -> ONNXTensorType:
        if node_type == "SC_POPCOUNT":
            return ONNXTensorType(elem_type=6, shape=shape)  # int32
        return ONNXTensorType(elem_type=9, shape=shape)  # bool for SC bitstreams

    def export(
        self,
        ir_graph: Any,
        input_shapes: Dict[str, Tuple[int, ...]],
        metadata: Dict[str, str] | None = None,
    ) -> ONNXGraph:
        """Convert SC-IR graph to ONNX graph representation."""
        from sc_neurocore.export.compiler_export import CompilerExporter

        exporter = CompilerExporter()
        sorted_nodes = exporter._topological_sort(ir_graph.nodes)

        graph = ONNXGraph(name=self.graph_name, metadata=metadata or {})

        # Register inputs
        for inp_name, shape in input_shapes.items():
            graph.inputs.append((inp_name, ONNXTensorType(elem_type=9, shape=shape)))

        # Track shapes for inference
        shapes: Dict[str, Tuple[int, ...]] = dict(input_shapes)

        # Convert nodes
        last_output = ""
        for node in sorted_nodes:
            op = SC_OP_MAP.get(node.type)
            if op is None:
                continue

            # Shape inference
            if node.type in ("SC_AND", "SC_MUX", "LIF_MEMBRANE"):
                out_shape = shapes.get(node.inputs[0], (1,))
            elif node.type == "SC_POPCOUNT":
                in_shape = shapes.get(node.inputs[0], (1,))
                out_shape = in_shape[:-1] + (1,) if len(in_shape) > 1 else (1,)
            else:
                out_shape = (1,)

            shapes[node.output] = out_shape

            # Build ONNX node
            attrs = {}
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

        # Register final output
        if last_output and last_output in shapes:
            graph.outputs.append(
                (last_output, self._infer_type("LIF_MEMBRANE", shapes[last_output]))
            )

        return graph
