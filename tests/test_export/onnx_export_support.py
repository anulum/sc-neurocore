# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_onnx_export.py

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


__all__ = [
    "json",
    "unittest",
    "Any",
    "pytest",
    "onnx_export",
    "ONNXExporter",
    "ONNXGraph",
    "ONNXNode",
    "ONNXTensorType",
    "SCPN_DOMAIN",
    "MockNode",
    "MockGraph",
    "simple_graph",
]
