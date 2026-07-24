# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFindFrequentItemsets from former test_spade_gpfa.py

"""Focused suite: TestFindFrequentItemsets from former test_spade_gpfa.py."""

from __future__ import annotations

from tests.spade_gpfa_support import *  # noqa: F403


class TestFindFrequentItemsets:
    def test_single_coactive_pair(self):
        mat = np.array(
            [
                [1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
            ],
            dtype=np.int8,
        )
        result = _find_frequent_itemsets(mat, min_support=3, max_size=3)
        pair_sets = [s for s, c in result if len(s) == 2]
        assert frozenset([0, 1]) in pair_sets

    def test_min_support_filters(self):
        mat = np.array(
            [
                [1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        result = _find_frequent_itemsets(mat, min_support=3, max_size=2)
        pair_sets = [s for s, c in result if len(s) == 2]
        assert len(pair_sets) == 0
