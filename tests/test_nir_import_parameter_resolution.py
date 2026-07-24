# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParameterResolution from former test_nir_import.py

"""Focused suite: TestParameterResolution from former test_nir_import.py."""

from __future__ import annotations

from tests.nir_import_support import *  # noqa: F403


class TestParameterResolution:
    def test_defaults_are_applied(self):
        g = _one("LIF")
        assert g.parameters["n0"]["tau"] == 20.0
        assert g.parameters["n0"]["v_threshold"] == 1.0

    def test_node_params_override_defaults(self):
        g = _one("LIF", tau=7.0, v_threshold=2.0)
        assert g.parameters["n0"]["tau"] == 7.0
        assert "7.0" in g.equations["n0"]
        assert g.thresholds["n0"] == "v > 2.0"

    def test_unknown_params_are_ignored(self):
        g = _one("LIF", not_a_param=99.0)
        assert "not_a_param" not in g.parameters["n0"]

    def test_distinct_time_constants_substituted_independently(self):
        # tau_syn and tau_mem must not clobber one another (longest-first).
        g = _one("CubaLIF", tau_syn=3.0, tau_mem=17.0)
        assert "3.0" in g.state_equations["n0"]["i_syn"]
        assert "17.0" in g.state_equations["n0"]["v"]
        assert "tau" not in g.state_equations["n0"]["v"]
