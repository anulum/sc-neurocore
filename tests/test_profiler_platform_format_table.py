# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormatTable from former test_profiler_platform.py

"""Focused suite: TestFormatTable from former test_profiler_platform.py."""

from __future__ import annotations

from tests.profiler_platform_support import *  # noqa: F403

class TestFormatTable:
    def test_table_format(self):
        results = compare(layer_sizes=[(16, 8)], platforms=["python"])
        table = format_table(results)
        assert "Platform" in table
        assert "python" in table
        assert "(ms)" in table

    def test_unavailable_in_table(self):
        r = PlatformResult(
            platform="missing",
            latency_ms=0,
            throughput_inf_per_s=0,
            power_mw=0,
            energy_per_inf_nj=0,
            available=False,
            notes="N/A",
        )
        table = format_table([r])
        assert "N/A" in table
