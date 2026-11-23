# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_photonic_emitter.py

"""Graph stand-ins shared by the photonic emitter tests."""

from __future__ import annotations

import unittest
from sc_neurocore.optics.photonic_emitter import PhotonicEmitter


class MockNode:
    """A graph node with a type, an id, input wires and an output wire."""

    def __init__(self, t: str, i: str, ins: list[str], out: str) -> None:
        self.type, self.id, self.inputs, self.output = t, i, ins, out


class MockGraph:
    """A graph that holds only its list of nodes."""

    def __init__(self, nodes: list[MockNode]) -> None:
        self.nodes = nodes


__all__ = ["unittest", "PhotonicEmitter", "MockNode", "MockGraph"]
