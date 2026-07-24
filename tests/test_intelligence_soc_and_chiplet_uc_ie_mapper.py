# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUCIeMapper from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestUCIeMapper from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403


class TestUCIeMapper:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import map_ucie_protocol

        r = map_ucie_protocol({"core_a": 64, "core_b": 128})
        assert r.lanes["core_a"] >= 1
        assert r.lanes["core_b"] >= 1
        assert r.total_bandwidth_gbps > 0
        assert "UCIe" in r.protocol_version
