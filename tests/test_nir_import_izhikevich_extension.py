# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIzhikevichExtension from former test_nir_import.py

"""Focused suite: TestIzhikevichExtension from former test_nir_import.py."""

from __future__ import annotations

from tests.nir_import_support import *  # noqa: F403

class TestIzhikevichExtension:
    def test_izhikevich_full_model(self):
        g = _one("Izhikevich")
        assert set(g.state_equations["n0"]) == {"u", "v"}
        assert "0.04" in g.equations["n0"]
        assert g.thresholds["n0"] == "v > 30"
        assert g.resets["n0"] == "v = -65.0; u = u + 8.0"

    def test_izhikevich_params_override(self):
        g = _one("izh", a=0.1, b=0.25)
        assert g.parameters["n0"]["a"] == 0.1 and g.parameters["n0"]["b"] == 0.25
        assert "u = u + 8.0" in g.resets["n0"]
