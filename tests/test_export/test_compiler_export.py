# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler Export Tests

"""Tests for the SSA compiler export module."""

from __future__ import annotations

import unittest

from sc_neurocore.export.compiler_export import CompilerExporter, SSAEnvironment, ShapeInference


class MockNode:
    def __init__(self, t, i, ins, out, **kwargs):
        self.type = t
        self.id = i
        self.inputs = ins
        self.output = out
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class TestSSAEnvironment(unittest.TestCase):
    def test_allocate_sequential(self):
        ssa = SSAEnvironment()
        r0 = ssa.allocate("a")
        r1 = ssa.allocate("b")
        self.assertEqual(r0, "%0")
        self.assertEqual(r1, "%1")

    def test_get_allocated(self):
        ssa = SSAEnvironment()
        ssa.allocate("x")
        self.assertEqual(ssa.get("x"), "%0")

    def test_get_unallocated_returns_global(self):
        ssa = SSAEnvironment()
        self.assertEqual(ssa.get("input_a"), "%input_a")


class TestShapeInference(unittest.TestCase):
    def test_and_preserves_shape(self):
        si = ShapeInference({"a": (128, 1024), "b": (128, 1024)})
        node = MockNode("SC_AND", "m0", ["a", "b"], "out")
        si.infer(node)
        self.assertEqual(si.shapes["out"], (128, 1024))

    def test_popcount_reduces_last_dim(self):
        si = ShapeInference({"a": (128, 1024)})
        node = MockNode("SC_POPCOUNT", "p0", ["a"], "out")
        si.infer(node)
        self.assertEqual(si.shapes["out"], (128, 1))

    def test_lif_preserves_shape(self):
        si = ShapeInference({"a": (64, 512)})
        node = MockNode("LIF_MEMBRANE", "n0", ["a"], "out")
        si.infer(node)
        self.assertEqual(si.shapes["out"], (64, 512))


class TestTopologicalSort(unittest.TestCase):
    def test_forward_order(self):
        exporter = CompilerExporter()
        nodes = [
            MockNode("SC_AND", "m1", ["a", "b"], "c"),
            MockNode("LIF_MEMBRANE", "n1", ["c"], "d"),
        ]
        sorted_nodes = exporter._topological_sort(nodes)
        self.assertEqual([n.id for n in sorted_nodes], ["m1", "n1"])

    def test_reverse_input_order(self):
        exporter = CompilerExporter()
        nodes = [
            MockNode("LIF_MEMBRANE", "n1", ["c"], "d"),
            MockNode("SC_AND", "m1", ["a", "b"], "c"),
        ]
        sorted_nodes = exporter._topological_sort(nodes)
        self.assertEqual([n.id for n in sorted_nodes], ["m1", "n1"])

    def test_cycle_detection(self):
        exporter = CompilerExporter()
        nodes = [
            MockNode("SC_AND", "m1", ["d"], "c"),
            MockNode("SC_AND", "m2", ["c"], "d"),
        ]
        with self.assertRaises(ValueError):
            exporter._topological_sort(nodes)

    def test_independent_nodes(self):
        exporter = CompilerExporter()
        nodes = [
            MockNode("SC_AND", "m1", ["a", "b"], "c"),
            MockNode("SC_AND", "m2", ["x", "y"], "z"),
        ]
        sorted_nodes = exporter._topological_sort(nodes)
        self.assertEqual(len(sorted_nodes), 2)


class TestMLIRExport(unittest.TestCase):
    def test_basic_export(self):
        exporter = CompilerExporter()
        nodes = [
            MockNode("SC_AND", "m1", ["input_a", "input_b"], "mac_1"),
            MockNode("LIF_MEMBRANE", "n1", ["mac_1"], "spike_out", threshold=0.75, leak=0.9),
        ]
        inputs = {"input_a": (128, 1024), "input_b": (128, 1024)}
        mlir = exporter.export_to_mlir(MockGraph(nodes), inputs)

        self.assertIn("module {", mlir)
        self.assertIn("func.func @sc_network_forward", mlir)
        self.assertIn("scpn.and", mlir)
        self.assertIn("scpn.lif", mlir)
        self.assertIn("return", mlir)

    def test_popcount_type(self):
        exporter = CompilerExporter()
        nodes = [
            MockNode("SC_POPCOUNT", "p1", ["input_a"], "count"),
        ]
        inputs = {"input_a": (128, 1024)}
        mlir = exporter.export_to_mlir(MockGraph(nodes), inputs)
        self.assertIn("scpn.popcount", mlir)
        self.assertIn("i32", mlir)

    def test_mux_export(self):
        exporter = CompilerExporter()
        nodes = [
            MockNode("SC_MUX", "x1", ["a", "b", "sel"], "out"),
        ]
        inputs = {"a": (64,), "b": (64,), "sel": (64,)}
        mlir = exporter.export_to_mlir(MockGraph(nodes), inputs)
        self.assertIn("scpn.mux", mlir)

    def test_large_graph(self):
        exporter = CompilerExporter()
        nodes = []
        for i in range(100):
            nodes.append(MockNode("SC_AND", f"m{i}", [f"a{i}", f"b{i}"], f"r{i}"))
        inputs = {f"a{i}": (128,) for i in range(100)}
        inputs.update({f"b{i}": (128,) for i in range(100)})
        mlir = exporter.export_to_mlir(MockGraph(nodes), inputs)
        self.assertEqual(mlir.count("scpn.and"), 100)


class TestMlirTypeFormatting(unittest.TestCase):
    def test_scalar_shapes_use_the_bare_element_type(self):
        exporter = CompilerExporter()
        self.assertEqual(exporter._format_mlir_type((), "i1"), "i1")
        self.assertEqual(exporter._format_mlir_type((1,), "i8"), "i8")

    def test_multidimensional_shapes_use_a_tensor_type(self):
        exporter = CompilerExporter()
        self.assertEqual(exporter._format_mlir_type((2, 3), "i1"), "tensor<2x3xi1>")


if __name__ == "__main__":
    unittest.main()
