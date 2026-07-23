# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalPropertyLink from former test_explainability.py

"""Focused suite: TestFormalPropertyLink from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403

class TestFormalPropertyLink:
    def test_default_fields(self):
        fp = FormalPropertyLink(property_name="no_metastability")
        assert fp.status == "unverified"
        assert fp.engine == "sby"

    def test_verify_with_formal_props(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20)
        props = [
            FormalPropertyLink("no_metastability", status="proven", bounded_depth=20),
            FormalPropertyLink("lfsr_period", status="proven", bounded_depth=65535),
        ]
        report = engine.verify(formal_properties=props)
        assert len(report.formal_properties) == 2
        assert report.formal_properties[0].status == "proven"
