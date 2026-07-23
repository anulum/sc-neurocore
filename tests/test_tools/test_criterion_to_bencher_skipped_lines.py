# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSkippedLines from former test_criterion_to_bencher.py

"""Focused suite: TestSkippedLines from former test_criterion_to_bencher.py."""

from __future__ import annotations

from criterion_to_bencher_support import *  # noqa: F403

class TestSkippedLines:
    """Lines that are not well-formed measurements yield nothing."""

    def test_time_line_without_bracket_is_skipped(self) -> None:
        assert _one("noisy  time:   pending") is None

    def test_bracket_with_single_estimate_is_skipped(self) -> None:
        assert _one("weird  time:   [42.0 µs]") is None

    def test_warmup_and_progress_lines_are_ignored(self) -> None:
        text = (
            "Benchmarking vip_1k_steps: Warming up for 3.0000 s\n"
            "Benchmarking vip_1k_steps: Collecting 100 samples in estimated 5.01 s\n"
            "vip_1k_steps  time:   [999.50 µs 1.0001 ms 1.0050 ms]\n"
            "Found 3 outliers among 100 measurements (3.00%)"
        )
        assert list(_CONVERTER.convert(text)) == [
            "test vip_1k_steps ... bench: 1000100 ns/iter (+/- 0)"
        ]
