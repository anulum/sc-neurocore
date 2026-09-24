# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNodeTypes from former test_nir_import.py

"""Focused suite: TestNodeTypes from former test_nir_import.py."""

from __future__ import annotations

from tests.nir_import_support import *  # noqa: F403


class TestNodeTypes:
    @pytest.mark.parametrize("missing", [None, "", "  ", 7])
    def test_a_node_without_a_type_is_refused(self, missing):
        spec = {} if missing is None else {"type": missing}
        with pytest.raises(ValueError, match="NIR node 'n0' has no type"):
            import_nir_graph({"nodes": {"n0": spec}, "edges": []})

    def test_if_has_threshold_no_leak(self):
        g = _one("IF")
        assert g.thresholds["n0"] == "v > 1.0"
        assert "tau" not in g.equations["n0"]

    def test_li_has_no_threshold(self):
        g = _one("LI", tau=15)
        assert g.thresholds["n0"] is None and g.resets["n0"] is None
        assert "15.0" in g.equations["n0"]

    def test_integrator_pure(self):
        g = _one("I")
        assert g.node_types["n0"] == "integrator"
        assert g.thresholds["n0"] is None

    def test_cuba_lif_is_two_compartment(self):
        g = _one("CubaLIF")
        assert set(g.state_equations["n0"]) == {"i_syn", "v"}
        assert g.thresholds["n0"] == "v > 1.0"
        # membrane equation is the one exposed flat
        assert g.equations["n0"] == g.state_equations["n0"]["v"]

    def test_cuba_li_has_no_threshold(self):
        g = _one("CubaLI")
        assert set(g.state_equations["n0"]) == {"i_syn", "v"}
        assert g.thresholds["n0"] is None
