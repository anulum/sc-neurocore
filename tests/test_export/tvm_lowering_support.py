# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_tvm_lowering.py

"""Mock SC-IR nodes and graphs shared by the TVM lowering tests."""

from __future__ import annotations

import unittest
from typing import Any

from sc_neurocore.export.tvm_lowering import (
    TVMLowering,
    TargetSchedule,
    TargetDevice,
    RelayFunction,
)


class MockNode:
    """An SC-IR node with a type, an id, input edges, an output edge and attributes."""

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
    """An SC-IR graph holding its nodes in order."""

    def __init__(self, nodes: list[MockNode]) -> None:
        self.nodes = nodes


def simple_graph() -> MockGraph:
    """Return an AND node feeding a LIF neuron with threshold 0.75 and leak 0.9."""
    return MockGraph(
        [
            MockNode("SC_AND", "m1", ["input_a", "input_b"], "mac_1"),
            MockNode("LIF_MEMBRANE", "n1", ["mac_1"], "spike_out", threshold=0.75, leak=0.9),
        ]
    )


__all__ = [
    "unittest",
    "TVMLowering",
    "TargetSchedule",
    "TargetDevice",
    "RelayFunction",
    "MockNode",
    "MockGraph",
    "simple_graph",
]
