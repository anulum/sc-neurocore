# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestONNXTensorType from former test_onnx_export.py

"""Focused suite: TestONNXTensorType from former test_onnx_export.py."""

from __future__ import annotations

from onnx_export_support import *  # noqa: F403


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
