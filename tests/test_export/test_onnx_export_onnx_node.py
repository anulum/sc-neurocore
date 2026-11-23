# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestONNXNode from former test_onnx_export.py

"""Focused suite: TestONNXNode from former test_onnx_export.py."""

from __future__ import annotations

import unittest
from sc_neurocore.export.onnx_export import (
    ONNXNode,
    SCPN_DOMAIN,
)


class TestONNXNode(unittest.TestCase):
    """Dictionary form of ONNX nodes."""

    def test_to_dict_no_attrs(self) -> None:
        """A node without attributes has no attribute entry."""
        n = ONNXNode("ScAnd", SCPN_DOMAIN, ["a", "b"], ["c"], "and_1")
        d = n.to_dict()
        self.assertEqual(d["op_type"], "ScAnd")
        self.assertNotIn("attribute", d)

    def test_to_dict_with_attrs(self) -> None:
        """A node with two attributes lists both."""
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
