# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTopologicalSort from former test_photonic_emitter.py

"""Focused suite: TestTopologicalSort from former test_photonic_emitter.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from photonic_emitter_support import *  # noqa: F403

class TestTopologicalSort(unittest.TestCase):
    def test_forward_order(self):
        emitter = PhotonicEmitter()
        nodes = [
            MockNode("SC_AND", "m1", ["a", "b"], "c"),
            MockNode("LIF_MEMBRANE", "n1", ["c"], "d"),
        ]
        sorted_nodes = emitter._topological_sort(nodes)
        self.assertEqual([n.id for n in sorted_nodes], ["m1", "n1"])

    def test_reverse_order(self):
        emitter = PhotonicEmitter()
        nodes = [
            MockNode("LIF_MEMBRANE", "n1", ["c"], "d"),
            MockNode("SC_AND", "m1", ["a", "b"], "c"),
        ]
        sorted_nodes = emitter._topological_sort(nodes)
        self.assertEqual([n.id for n in sorted_nodes], ["m1", "n1"])
