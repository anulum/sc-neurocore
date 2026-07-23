# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegulatoryMetadata from former test_explainability.py

"""Focused suite: TestRegulatoryMetadata from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403

class TestRegulatoryMetadata:
    def test_default_fields(self):
        rm = RegulatoryMetadata()
        assert rm.device_class == "Class II"
        assert rm.review_status == "pending"

    def test_verify_with_regulatory(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20)
        reg = RegulatoryMetadata(
            device_class="Class III",
            intended_use="BCI motor cortex",
            software_version="3.12.0",
        )
        report = engine.verify(regulatory=reg)
        assert report.regulatory is not None
        assert report.regulatory.device_class == "Class III"
