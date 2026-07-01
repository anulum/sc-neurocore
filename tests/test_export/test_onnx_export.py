# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ONNX Export Tests

from __future__ import annotations

import json
import unittest
from typing import Any

import pytest

import sc_neurocore.export.onnx_export as onnx_export
from sc_neurocore.export.onnx_export import (
    ONNXExporter,
    ONNXGraph,
    ONNXNode,
    ONNXTensorType,
    SCPN_DOMAIN,
)


class MockNode:
    def __init__(
        self,
        t: str,
        i: str,
        ins: list[str],
        out: str,
        **kwargs: Any,
    ) -> None:
        self.type, self.id, self.inputs, self.output = t, i, ins, out
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockGraph:
    def __init__(self, nodes: list[MockNode]) -> None:
        self.nodes = nodes


def simple_graph() -> MockGraph:
    return MockGraph(
        [
            MockNode("SC_AND", "m1", ["input_a", "input_b"], "mac_1"),
            MockNode("LIF_MEMBRANE", "n1", ["mac_1"], "spike_out", threshold=0.75, leak=0.9),
        ]
    )


class TestONNXTensorType(unittest.TestCase):
    def test_to_dict(self) -> None:
        tt = ONNXTensorType(elem_type=9, shape=(128, 1024))
        d = tt.to_dict()
        self.assertEqual(d["elem_type"], 9)
        self.assertEqual(len(d["shape"]["dim"]), 2)

    def test_scalar_shape(self) -> None:
        tt = ONNXTensorType(elem_type=1, shape=(1,))
        d = tt.to_dict()
        self.assertEqual(d["shape"]["dim"][0]["dim_value"], 1)


class TestONNXNode(unittest.TestCase):
    def test_to_dict_no_attrs(self) -> None:
        n = ONNXNode("ScAnd", SCPN_DOMAIN, ["a", "b"], ["c"], "and_1")
        d = n.to_dict()
        self.assertEqual(d["op_type"], "ScAnd")
        self.assertNotIn("attribute", d)

    def test_to_dict_with_attrs(self) -> None:
        n = ONNXNode(
            "LifNeuron",
            SCPN_DOMAIN,
            ["a"],
            ["b"],
            "lif_1",
            attributes={"threshold": 0.75, "leak": 0.9},
        )
        d = n.to_dict()
        self.assertIn("attribute", d)
        self.assertEqual(len(d["attribute"]), 2)


class TestONNXExporter(unittest.TestCase):
    def test_export_produces_graph(self) -> None:
        exp = ONNXExporter()
        graph = exp.export(simple_graph(), {"input_a": (128, 1024), "input_b": (128, 1024)})
        self.assertIsInstance(graph, ONNXGraph)

    def test_graph_has_nodes(self) -> None:
        exp = ONNXExporter()
        graph = exp.export(simple_graph(), {"input_a": (128, 1024), "input_b": (128, 1024)})
        self.assertEqual(len(graph.nodes), 2)

    def test_graph_has_inputs(self) -> None:
        exp = ONNXExporter()
        graph = exp.export(simple_graph(), {"input_a": (128, 1024), "input_b": (128, 1024)})
        self.assertEqual(len(graph.inputs), 2)

    def test_graph_has_output(self) -> None:
        exp = ONNXExporter()
        graph = exp.export(simple_graph(), {"input_a": (128, 1024), "input_b": (128, 1024)})
        self.assertGreaterEqual(len(graph.outputs), 1)
        self.assertEqual(graph.outputs[0][0], "spike_out")

    def test_custom_domain(self) -> None:
        exp = ONNXExporter()
        graph = exp.export(simple_graph(), {"input_a": (128,), "input_b": (128,)})
        for node in graph.nodes:
            self.assertEqual(node.domain, SCPN_DOMAIN)

    def test_lif_attributes_preserved(self) -> None:
        exp = ONNXExporter()
        graph = exp.export(simple_graph(), {"input_a": (128,), "input_b": (128,)})
        lif_node = [n for n in graph.nodes if n.op_type == "LifNeuron"][0]
        self.assertAlmostEqual(lif_node.attributes["threshold"], 0.75)
        self.assertAlmostEqual(lif_node.attributes["leak"], 0.9)

    def test_to_json_valid(self) -> None:
        exp = ONNXExporter()
        graph = exp.export(simple_graph(), {"input_a": (128,), "input_b": (128,)})
        j = graph.to_json()
        parsed = json.loads(j)
        self.assertIn("graph", parsed)
        self.assertIn("opset_import", parsed)

    def test_metadata_propagated(self) -> None:
        exp = ONNXExporter()
        graph = exp.export(
            simple_graph(),
            {"input_a": (128,), "input_b": (128,)},
            metadata={"bitstream_length": "256", "target": "Xilinx_ZU9EG"},
        )
        d = graph.to_dict()
        keys = {p["key"] for p in d["metadata_props"]}
        self.assertIn("bitstream_length", keys)
        self.assertIn("target", keys)

    def test_popcount_node(self) -> None:
        graph = MockGraph(
            [
                MockNode("SC_POPCOUNT", "pc1", ["input_a"], "count_out"),
            ]
        )
        exp = ONNXExporter()
        result = exp.export(graph, {"input_a": (128, 1024)})
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.nodes[0].op_type, "ScPopcount")

    def test_popcount_final_output_is_int32_tensor(self) -> None:
        graph = MockGraph(
            [
                MockNode("SC_POPCOUNT", "pc1", ["input_a"], "count_out"),
            ]
        )
        exp = ONNXExporter()
        result = exp.export(graph, {"input_a": (128, 1024)})

        self.assertEqual(result.outputs, [("count_out", ONNXTensorType(elem_type=6, shape=(128, 1)))])

    def test_unknown_nodes_are_skipped_without_becoming_outputs(self) -> None:
        graph = MockGraph(
            [
                MockNode("UNKNOWN", "u1", ["input_a"], "unknown_out"),
                MockNode("SC_AND", "m1", ["input_a", "input_b"], "and_out"),
            ]
        )
        exp = ONNXExporter()
        result = exp.export(graph, {"input_a": (8,), "input_b": (8,)})

        self.assertEqual([node.name for node in result.nodes], ["ScAnd_m1"])
        self.assertEqual(result.outputs[0][0], "and_out")

    def test_mapped_node_without_shape_rule_fails_closed(self) -> None:
        graph = MockGraph([MockNode("SC_CUSTOM", "c1", ["input_a"], "custom_out")])
        exp = ONNXExporter()
        onnx_export.SC_OP_MAP["SC_CUSTOM"] = "ScCustom"
        try:
            with pytest.raises(ValueError, match="No ONNX shape rule"):
                exp.export(graph, {"input_a": (8,)})
        finally:
            del onnx_export.SC_OP_MAP["SC_CUSTOM"]

    def test_empty_graph(self) -> None:
        graph = MockGraph([])
        exp = ONNXExporter()
        result = exp.export(graph, {"input_a": (128,)})
        self.assertEqual(len(result.nodes), 0)


if __name__ == "__main__":
    unittest.main()
