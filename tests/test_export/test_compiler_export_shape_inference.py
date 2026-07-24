# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestShapeInference from former test_compiler_export.py

"""Focused suite: TestShapeInference from former test_compiler_export.py."""

from __future__ import annotations

from compiler_export_support import *  # noqa: F403


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
