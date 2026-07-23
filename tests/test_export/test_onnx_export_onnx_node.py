# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestONNXNode from former test_onnx_export.py

"""Focused suite: TestONNXNode from former test_onnx_export.py."""

from __future__ import annotations

from onnx_export_support import *  # noqa: F403

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
