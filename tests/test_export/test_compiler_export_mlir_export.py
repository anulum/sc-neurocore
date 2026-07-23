# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMLIRExport from former test_compiler_export.py

"""Focused suite: TestMLIRExport from former test_compiler_export.py."""

from __future__ import annotations

from compiler_export_support import *  # noqa: F403

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
