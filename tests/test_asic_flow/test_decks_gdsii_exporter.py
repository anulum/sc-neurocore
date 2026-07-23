# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGDSIIExporter from former test_decks.py

"""Focused suite: TestGDSIIExporter from former test_decks.py."""

from __future__ import annotations

from tests.test_asic_flow.decks_support import *  # noqa: F403

class TestGDSIIExporter:
    def test_open_source_export(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = GDSIIExporter.generate(pdk, design)
        assert ".gds" in script
        assert design.top_module in script

    def test_commercial_export(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.INTEL16)
        design = DesignParams()
        script = GDSIIExporter.generate(pdk, design)
        assert "vendor" in script.lower()
