# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_tvm_lowering.py

from __future__ import annotations

import unittest
from sc_neurocore.export.tvm_lowering import (
    TVMLowering,
    TargetSchedule,
    TargetDevice,
    RelayFunction,
)


class MockNode:
    def __init__(self, t, i, ins, out, **kwargs):
        self.type, self.id, self.inputs, self.output = t, i, ins, out
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockGraph:
    def __init__(self, nodes):
        self.nodes = nodes


def simple_graph():
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
