# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTopologicalSort from former test_compiler_export.py

"""Focused suite: TestTopologicalSort from former test_compiler_export.py."""

from __future__ import annotations

from compiler_export_support import *  # noqa: F403

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
