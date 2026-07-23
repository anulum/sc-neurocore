# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUnitBoundaryStraddle from former test_criterion_to_bencher.py

"""Focused suite: TestUnitBoundaryStraddle from former test_criterion_to_bencher.py."""

from __future__ import annotations

from criterion_to_bencher_support import *  # noqa: F403

class TestUnitBoundaryStraddle:
    """The regression the missing test let through: median paired with the wrong unit."""

    def test_us_to_ms_straddle_median_uses_its_own_unit(self) -> None:
        # Median 1.0001 ms must scale as ms (1000100 ns), NOT as the low bound's µs (1000 ns).
        assert _one("vip_1k_steps  time:   [999.50 µs 1.0001 ms 1.0050 ms]") == 1_000_100

    def test_ns_to_us_straddle(self) -> None:
        assert _one("k  time:   [980.0 ns 1.0002 µs 1.0100 µs]") == 1_000  # 1.0002 µs → 1000 ns

    def test_ms_to_s_straddle(self) -> None:
        assert _one("k  time:   [999.0 ms 1.0005 s 1.0100 s]") == 1_000_500_000
