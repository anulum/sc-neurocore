# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeSorter from former test_symbolic.py

"""Focused suite: TestSpikeSorter from former test_symbolic.py."""

from __future__ import annotations

from tests.symbolic_support import *  # noqa: F403


class TestSpikeSorter:
    def test_sort_basic(self):
        assert spike_sort([3, 1, 4, 1, 5, 9, 2, 6]) == sorted([3, 1, 4, 1, 5, 9, 2, 6])

    def test_sort_already_sorted(self):
        assert spike_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_sort_reversed(self):
        assert spike_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_sort_single(self):
        assert spike_sort([42]) == [42]

    def test_sort_empty(self):
        assert spike_sort([]) == []

    def test_sort_duplicates(self):
        assert spike_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_sort_large_values(self):
        vals = [200, 50, 128, 0, 255]
        assert spike_sort(vals) == sorted(vals)
