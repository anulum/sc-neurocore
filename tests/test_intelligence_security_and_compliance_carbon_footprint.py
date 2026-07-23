# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCarbonFootprint from former test_intelligence_security_and_compliance.py

"""Focused suite: TestCarbonFootprint from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403

class TestCarbonFootprint:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import estimate_carbon_footprint

        c = estimate_carbon_footprint("artix7")
        assert c.manufacturing_kg_co2 > 0
        assert c.total_5yr_kg_co2 > c.manufacturing_kg_co2

    def test_biological_low(self):
        from sc_neurocore.compiler.intelligence import estimate_carbon_footprint

        c = estimate_carbon_footprint("finalspark_neuroplatform")
        assert c.manufacturing_kg_co2 <= 0.5
