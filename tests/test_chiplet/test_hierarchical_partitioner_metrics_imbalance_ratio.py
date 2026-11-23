# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImbalanceRatio from former test_hierarchical_partitioner_metrics.py

"""Focused suite: TestImbalanceRatio from former test_hierarchical_partitioner_metrics.py."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    calculate_imbalance_ratio,
)


class TestImbalanceRatio:
    """Imbalance ratio of the partition sizes."""

    def test_perfect_balance(self) -> None:
        """Equal partition sizes give an imbalance ratio of zero."""
        parts = [[0, 1], [2, 3], [4, 5]]
        assert calculate_imbalance_ratio(parts) == 0.0

    def test_imbalanced(self) -> None:
        """Unequal partition sizes give a positive imbalance ratio."""
        parts = [[0, 1, 2, 3], [4]]
        ratio = calculate_imbalance_ratio(parts)
        assert ratio > 0.0

    def test_empty(self) -> None:
        """No partitions at all give an imbalance ratio of zero."""
        assert calculate_imbalance_ratio([]) == 0.0

    def test_single_partition(self) -> None:
        """One partition gives an imbalance ratio of zero."""
        assert calculate_imbalance_ratio([[0, 1, 2]]) == 0.0
