# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSingleUnitEstimates from former test_criterion_to_bencher.py

"""Focused suite: TestSingleUnitEstimates from former test_criterion_to_bencher.py."""

from __future__ import annotations

from criterion_to_bencher_support import *  # noqa: F403

class TestSingleUnitEstimates:
    """Non-straddle triplets in each unit convert on the median."""

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("sst_1k_steps  time:   [481.99 µs 482.09 µs 482.91 µs]", 482_090),
            ("adex_1k_steps  time:   [29.9 µs 30.0 µs 30.1 µs]", 30_000),
            ("dense  time:   [25.900 ms 26.075 ms 26.120 ms]", 26_075_000),
            ("fast  time:   [11.0 ns 12.0 ns 13.0 ns]", 12),
            ("slow  time:   [4.9 s 5.0 s 5.1 s]", 5_000_000_000),
        ],
    )
    def test_median_conversion(self, line: str, expected: int) -> None:
        assert _one(line) == expected
