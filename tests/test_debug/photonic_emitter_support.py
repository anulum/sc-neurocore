# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_photonic_emitter.py

from __future__ import annotations

"""Tests for the photonic emitter module."""
import unittest
from sc_neurocore.optics.photonic_emitter import PhotonicEmitter


class MockNode:
    def __init__(self, t, i, ins, out):
        self.type, self.id, self.inputs, self.output = t, i, ins, out


class MockGraph:
    def __init__(self, nodes):
        self.nodes = nodes


__all__ = ["unittest", "PhotonicEmitter", "MockNode", "MockGraph"]
