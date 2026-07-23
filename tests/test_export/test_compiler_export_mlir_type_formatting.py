# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMlirTypeFormatting from former test_compiler_export.py

"""Focused suite: TestMlirTypeFormatting from former test_compiler_export.py."""

from __future__ import annotations

from compiler_export_support import *  # noqa: F403

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
