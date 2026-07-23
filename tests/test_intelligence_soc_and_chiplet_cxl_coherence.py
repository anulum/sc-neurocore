# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCXLCoherence from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestCXLCoherence from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403

class TestCXLCoherence:
    """CXL.mem Type-3 device mapping."""

    def test_basic_mapping(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        m = advise_cxl_mapping(10000, 1000000)
        assert m.device_count >= 1
        assert len(m.state_device_ids) >= 1
        assert len(m.weight_device_ids) >= 1
        assert m.total_capacity_gb > 0

    def test_streaming_uses_cxl_mem(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        m = advise_cxl_mapping(1000, 10000, access_pattern="streaming")
        assert m.coherence_protocol == "CXL.mem"

    def test_random_uses_cxl_cache(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        m = advise_cxl_mapping(1000, 10000, access_pattern="random")
        assert m.coherence_protocol == "CXL.cache"

    def test_respects_device_limit(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        m = advise_cxl_mapping(
            1000000000,
            10000000000,
            max_devices=4,
        )
        assert m.device_count <= 4

    def test_random_needs_more_bandwidth(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        s = advise_cxl_mapping(10000, 1000000, access_pattern="streaming")
        r = advise_cxl_mapping(10000, 1000000, access_pattern="random")
        assert r.host_bandwidth_gbps > s.host_bandwidth_gbps
