# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeSort from former test_spike_alu.py

"""Focused suite: TestSpikeSort from former test_spike_alu.py."""

from __future__ import annotations

from tests.spike_alu_support import *  # noqa: F403


class TestSpikeSort:
    def test_empty(self):
        assert spike_sort([]) == []

    def test_single(self):
        assert spike_sort([42]) == [42]

    def test_sorted_input(self):
        assert spike_sort([1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_reversed_input(self):
        assert spike_sort([9, 7, 5, 3, 1]) == [1, 3, 5, 7, 9]

    def test_duplicates(self):
        assert spike_sort([5, 5, 5]) == [5, 5, 5]

    def test_large_values(self):
        arr = [255, 0, 128, 64, 192]
        assert spike_sort(arr) == sorted(arr)

    def test_random_array(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=15).tolist()
        assert spike_sort(arr) == sorted(arr)
