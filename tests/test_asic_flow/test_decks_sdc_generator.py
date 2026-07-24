# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSDCGenerator from former test_decks.py

"""Focused suite: TestSDCGenerator from former test_decks.py."""

from __future__ import annotations

from tests.test_asic_flow.decks_support import *  # noqa: F403


class TestSDCGenerator:
    def test_generates_sdc(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        sdc = SDCGenerator.generate(pdk, design)
        assert "create_clock" in sdc
        assert design.clock_name in sdc

    def test_clock_period(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(target_frequency_mhz=100.0)
        sdc = SDCGenerator.generate(pdk, design)
        assert "10.000" in sdc

    def test_false_path_reset(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(reset_name="rst_n")
        sdc = SDCGenerator.generate(pdk, design)
        assert "rst_n" in sdc
        assert "false_path" in sdc

    def test_sc_fanout_constraint(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(sc_optimisation=SCASICOptimisationConfig(max_fanout=8))
        sdc = SDCGenerator.generate(pdk, design)
        assert "set_max_fanout 8" in sdc
