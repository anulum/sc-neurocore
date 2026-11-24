# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeSort from former test_spike_alu.py

"""Focused suite: TestSpikeSort from former test_spike_alu.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.symbolic.spike_logic import (
    spike_sort,
)


class TestSpikeSort:
    """Sorting values with spikes, across edge cases."""

    def test_empty(self) -> None:
        """Sorting an empty list gives an empty list."""
        assert spike_sort([]) == []

    def test_single(self) -> None:
        """Sorting one value returns it unchanged."""
        assert spike_sort([42]) == [42]

    def test_sorted_input(self) -> None:
        """An already sorted list stays sorted."""
        assert spike_sort([1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_reversed_input(self) -> None:
        """A reversed list is sorted ascending."""
        assert spike_sort([9, 7, 5, 3, 1]) == [1, 3, 5, 7, 9]

    def test_duplicates(self) -> None:
        """Duplicate values are kept."""
        assert spike_sort([5, 5, 5]) == [5, 5, 5]

    def test_large_values(self) -> None:
        """Large values sort as Python sorts them."""
        arr = [255, 0, 128, 64, 192]
        assert spike_sort(arr) == sorted(arr)

    def test_random_array(self) -> None:
        """A random array sorts as Python sorts it."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=15).tolist()
        assert spike_sort(arr) == sorted(arr)
