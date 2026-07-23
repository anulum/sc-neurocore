# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSharedTemplates from former test_nir_import.py

"""Focused suite: TestSharedTemplates from former test_nir_import.py."""

from __future__ import annotations

from tests.nir_import_support import *  # noqa: F403

class TestSharedTemplates:
    def test_equations_come_from_the_authoritative_bridge_table(self):
        # The importer must not carry its own divergent LIF dynamics: the leak
        # term and reset come straight from the shared template.
        g = _one("LIF", tau=20)
        assert g.equations["n0"] == "-(v - 0.0) / 20.0 + I * 1.0 / 20.0"
        assert g.thresholds["n0"] == "v > 1.0"
        assert g.resets["n0"] == "v = 0.0"

    def test_bridge_types_are_all_recognised(self):
        for canonical in NEURON_TEMPLATES:
            g = import_nir_graph({"nodes": {"n": {"type": canonical}}, "edges": []})
            assert g.node_types["n"] == canonical
