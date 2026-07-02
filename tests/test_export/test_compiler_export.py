# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler Export Tests

"""Tests for the SSA compiler export module."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from sc_neurocore.export.compiler_export import CompilerExporter, SSAEnvironment, ShapeInference


ShapeMap = dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class MockNode:
    """Typed SC-IR node fixture for compiler export tests."""

    type: str
    id: str
    inputs: tuple[str, ...]
    output: str
    threshold: float = 1.0
    leak: float = 0.9


@dataclass(frozen=True)
class MockGraph:
    """Typed SC-IR graph fixture exposing the public ``nodes`` contract."""

    nodes: tuple[MockNode, ...]


class TestSSAEnvironment(unittest.TestCase):
    """Verify SSA register allocation and external input lookup."""

    def test_allocate_sequential(self) -> None:
        ssa = SSAEnvironment()
        r0 = ssa.allocate("a")
        r1 = ssa.allocate("b")
        self.assertEqual(r0, "%0")
        self.assertEqual(r1, "%1")

    def test_get_allocated(self) -> None:
        ssa = SSAEnvironment()
        ssa.allocate("x")
        self.assertEqual(ssa.get("x"), "%0")

    def test_get_unallocated_returns_global(self) -> None:
        ssa = SSAEnvironment()
        self.assertEqual(ssa.get("input_a"), "%input_a")


class TestShapeInference(unittest.TestCase):
    """Verify shape propagation for supported SC-IR node types."""

    def test_and_preserves_shape(self) -> None:
        si = ShapeInference({"a": (128, 1024), "b": (128, 1024)})
        node = MockNode("SC_AND", "m0", ("a", "b"), "out")
        si.infer(node)
        self.assertEqual(si.shapes["out"], (128, 1024))

    def test_popcount_reduces_last_dim(self) -> None:
        si = ShapeInference({"a": (128, 1024)})
        node = MockNode("SC_POPCOUNT", "p0", ("a",), "out")
        si.infer(node)
        self.assertEqual(si.shapes["out"], (128, 1))

    def test_lif_preserves_shape(self) -> None:
        si = ShapeInference({"a": (64, 512)})
        node = MockNode("LIF_MEMBRANE", "n0", ("a",), "out")
        si.infer(node)
        self.assertEqual(si.shapes["out"], (64, 512))

    def test_missing_input_shape_fails_closed(self) -> None:
        si = ShapeInference({"a": (64,)})
        node = MockNode("SC_AND", "m0", ("a", "missing"), "out")
        with self.assertRaisesRegex(ValueError, "Missing input shape"):
            si.infer(node)


class TestTopologicalSort(unittest.TestCase):
    """Verify graph ordering rejects malformed dataflow contracts."""

    def test_forward_order(self) -> None:
        exporter = CompilerExporter()
        nodes = (
            MockNode("SC_AND", "m1", ("a", "b"), "c"),
            MockNode("LIF_MEMBRANE", "n1", ("c",), "d"),
        )
        sorted_nodes = exporter._topological_sort(nodes)
        self.assertEqual([n.id for n in sorted_nodes], ["m1", "n1"])

    def test_reverse_input_order(self) -> None:
        exporter = CompilerExporter()
        nodes = (
            MockNode("LIF_MEMBRANE", "n1", ("c",), "d"),
            MockNode("SC_AND", "m1", ("a", "b"), "c"),
        )
        sorted_nodes = exporter._topological_sort(nodes)
        self.assertEqual([n.id for n in sorted_nodes], ["m1", "n1"])

    def test_cycle_detection(self) -> None:
        exporter = CompilerExporter()
        nodes = (
            MockNode("SC_AND", "m1", ("d",), "c"),
            MockNode("SC_AND", "m2", ("c",), "d"),
        )
        with self.assertRaises(ValueError):
            exporter._topological_sort(nodes)

    def test_duplicate_node_ids_fail_closed(self) -> None:
        exporter = CompilerExporter()
        nodes = (
            MockNode("SC_AND", "m1", ("a", "b"), "c"),
            MockNode("SC_AND", "m1", ("x", "y"), "z"),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate node id"):
            exporter._topological_sort(nodes)

    def test_duplicate_outputs_fail_closed(self) -> None:
        exporter = CompilerExporter()
        nodes = (
            MockNode("SC_AND", "m1", ("a", "b"), "c"),
            MockNode("SC_AND", "m2", ("x", "y"), "c"),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate output edge"):
            exporter._topological_sort(nodes)

    def test_independent_nodes(self) -> None:
        exporter = CompilerExporter()
        nodes = (
            MockNode("SC_AND", "m1", ("a", "b"), "c"),
            MockNode("SC_AND", "m2", ("x", "y"), "z"),
        )
        sorted_nodes = exporter._topological_sort(nodes)
        self.assertEqual(len(sorted_nodes), 2)


class TestMLIRExport(unittest.TestCase):
    """Verify public MLIR export behavior for valid and invalid graphs."""

    def test_constructor_rejects_unsupported_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported compiler export target"):
            CompilerExporter(target="relay")

    def test_empty_graph_fails_closed(self) -> None:
        exporter = CompilerExporter()
        with self.assertRaisesRegex(ValueError, "at least one node"):
            exporter.export_to_mlir(MockGraph(()), {"input_a": (128,)})

    def test_unknown_node_type_fails_closed(self) -> None:
        exporter = CompilerExporter()
        nodes = (MockNode("SC_XOR", "x1", ("input_a", "input_b"), "out"),)
        inputs: ShapeMap = {"input_a": (128,), "input_b": (128,)}
        with self.assertRaisesRegex(ValueError, "Unsupported SC-IR node type"):
            exporter.export_to_mlir(MockGraph(nodes), inputs)

    def test_missing_external_shape_fails_closed(self) -> None:
        exporter = CompilerExporter()
        nodes = (MockNode("SC_AND", "m1", ("input_a", "input_b"), "out"),)
        with self.assertRaisesRegex(ValueError, "Missing input shape"):
            exporter.export_to_mlir(MockGraph(nodes), {"input_a": (128,)})

    def test_wrong_node_arity_fails_closed(self) -> None:
        exporter = CompilerExporter()
        nodes = (MockNode("SC_MUX", "x1", ("a", "b"), "out"),)
        inputs: ShapeMap = {"a": (64,), "b": (64,)}
        with self.assertRaisesRegex(ValueError, "expects 3 input"):
            exporter.export_to_mlir(MockGraph(nodes), inputs)

    def test_basic_export(self) -> None:
        exporter = CompilerExporter()
        nodes = (
            MockNode("SC_AND", "m1", ("input_a", "input_b"), "mac_1"),
            MockNode("LIF_MEMBRANE", "n1", ("mac_1",), "spike_out", threshold=0.75, leak=0.9),
        )
        inputs: ShapeMap = {"input_a": (128, 1024), "input_b": (128, 1024)}
        mlir = exporter.export_to_mlir(MockGraph(nodes), inputs)

        self.assertIn("module {", mlir)
        self.assertIn("func.func @sc_network_forward", mlir)
        self.assertIn("scpn.and", mlir)
        self.assertIn("scpn.lif", mlir)
        self.assertIn("return", mlir)

    def test_popcount_type(self) -> None:
        exporter = CompilerExporter()
        nodes = (MockNode("SC_POPCOUNT", "p1", ("input_a",), "count"),)
        inputs: ShapeMap = {"input_a": (128, 1024)}
        mlir = exporter.export_to_mlir(MockGraph(nodes), inputs)
        self.assertIn("scpn.popcount", mlir)
        self.assertIn("i32", mlir)

    def test_mux_export(self) -> None:
        exporter = CompilerExporter()
        nodes = (MockNode("SC_MUX", "x1", ("a", "b", "sel"), "out"),)
        inputs: ShapeMap = {"a": (64,), "b": (64,), "sel": (64,)}
        mlir = exporter.export_to_mlir(MockGraph(nodes), inputs)
        self.assertIn("scpn.mux", mlir)

    def test_large_graph(self) -> None:
        exporter = CompilerExporter()
        nodes = tuple(MockNode("SC_AND", f"m{i}", (f"a{i}", f"b{i}"), f"r{i}") for i in range(100))
        inputs: ShapeMap = {f"a{i}": (128,) for i in range(100)}
        inputs.update({f"b{i}": (128,) for i in range(100)})
        mlir = exporter.export_to_mlir(MockGraph(nodes), inputs)
        self.assertEqual(mlir.count("scpn.and"), 100)

    def test_scalar_input_signature_uses_bare_element_type(self) -> None:
        exporter = CompilerExporter()
        nodes = (MockNode("LIF_MEMBRANE", "n1", ("spike_in",), "spike_out"),)
        mlir = exporter.export_to_mlir(MockGraph(nodes), {"spike_in": ()})
        self.assertIn("func.func @sc_network_forward(%spike_in: i1)", mlir)
        self.assertIn("return %0 : i1", mlir)

    def test_invalid_input_names_fail_closed_before_emission(self) -> None:
        exporter = CompilerExporter()
        nodes = (MockNode("SC_AND", "m1", ("input-a", "input b"), "out"),)
        inputs: ShapeMap = {"input-a": (4,), "input b": (4,)}
        with self.assertRaisesRegex(ValueError, "Invalid input name"):
            exporter.export_to_mlir(MockGraph(nodes), inputs)

    def test_output_name_reuse_between_internal_and_input_edges_fails_closed(self) -> None:
        exporter = CompilerExporter()
        nodes = (MockNode("LIF_MEMBRANE", "n1", ("input_a",), "input_a"),)
        with self.assertRaisesRegex(ValueError, "collides with graph input"):
            exporter.export_to_mlir(MockGraph(nodes), {"input_a": (4,)})


class TestMlirTypeFormatting(unittest.TestCase):
    """Verify MLIR type rendering for scalar and tensor shapes."""

    def test_scalar_shapes_use_the_bare_element_type(self) -> None:
        exporter = CompilerExporter()
        self.assertEqual(exporter._format_mlir_type((), "i1"), "i1")
        self.assertEqual(exporter._format_mlir_type((1,), "i8"), "i8")

    def test_multidimensional_shapes_use_a_tensor_type(self) -> None:
        exporter = CompilerExporter()
        self.assertEqual(exporter._format_mlir_type((2, 3), "i1"), "tensor<2x3xi1>")

    def test_non_positive_tensor_dimension_fails_closed(self) -> None:
        exporter = CompilerExporter()
        with self.assertRaisesRegex(ValueError, "positive"):
            exporter._format_mlir_type((0,), "i1")


if __name__ == "__main__":
    unittest.main()
