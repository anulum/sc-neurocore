# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGraphStructure from former test_nir_import.py

"""Focused suite: TestGraphStructure from former test_nir_import.py."""

from __future__ import annotations

from tests.nir_import_support import *  # noqa: F403

class TestGraphStructure:
    def test_edges_and_framework(self):
        g = import_nir_graph(
            {"nodes": {"a": {"type": "LIF"}, "b": {"type": "LI"}}, "edges": [["a", "b"]]},
            framework="Norse",
        )
        assert ("a", "b") in g.edges
        assert g.framework == "Norse"
        assert set(g.node_types) == {"a", "b"}

    def test_empty_graph(self):
        g = import_nir_graph({})
        assert g.equations == {} and g.edges == []
