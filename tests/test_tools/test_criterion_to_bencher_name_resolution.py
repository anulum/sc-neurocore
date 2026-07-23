# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNameResolution from former test_criterion_to_bencher.py

"""Focused suite: TestNameResolution from former test_criterion_to_bencher.py."""

from __future__ import annotations

from criterion_to_bencher_support import *  # noqa: F403

class TestNameResolution:
    """The benchmark name comes from the result line or a preceding standalone line."""

    def test_name_on_same_line(self) -> None:
        assert list(_CONVERTER.convert("bench_x  time:   [1.0 µs 2.0 µs 3.0 µs]"))[0].startswith(
            "test bench_x ..."
        )

    def test_standalone_name_line(self) -> None:
        text = "bench_y\n                        time:   [1.0 µs 2.0 µs 3.0 µs]"
        out = list(_CONVERTER.convert(text))
        assert out == ["test bench_y ... bench: 2000 ns/iter (+/- 0)"]

    def test_change_line_does_not_become_name_or_emit(self) -> None:
        # A change: line must neither be captured as a name nor parsed as a measurement.
        text = "bench_z  time:   [1.0 µs 2.0 µs 3.0 µs]\n    change: [-1.0% +0.0% +1.0%]"
        out = list(_CONVERTER.convert(text))
        assert out == ["test bench_z ... bench: 2000 ns/iter (+/- 0)"]
