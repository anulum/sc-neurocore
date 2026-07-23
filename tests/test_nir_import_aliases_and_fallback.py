# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAliasesAndFallback from former test_nir_import.py

"""Focused suite: TestAliasesAndFallback from former test_nir_import.py."""

from __future__ import annotations

from tests.nir_import_support import *  # noqa: F403

class TestAliasesAndFallback:
    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("LeakyIntegrateAndFire", "lif"),
            ("integrate_and_fire", "if"),
            ("Leaky Integrator", "li"),
            ("cuba-lif", "cuba_lif"),
            ("CUBALI", "cuba_li"),
            ("Integrator", "integrator"),
        ],
    )
    def test_alias_resolution(self, alias, canonical):
        assert _one(alias).node_types["n0"] == canonical

    def test_unknown_type_falls_back_to_leaky_integrator(self):
        g = _one("Mystery", tau=5.0)
        assert g.node_types["n0"] == "li"
        assert "5.0" in g.equations["n0"]
