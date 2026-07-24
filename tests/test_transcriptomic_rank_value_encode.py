# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRankValueEncode from former test_transcriptomic.py

"""Focused suite: TestRankValueEncode from former test_transcriptomic.py."""

from __future__ import annotations

from tests.transcriptomic_support import *  # noqa: F403


class TestRankValueEncode:
    """Theodoris et al. (2023) rank-value tokenisation."""

    def test_all_zeros(self) -> None:
        result = rank_value_encode(np.zeros(10))
        assert len(result) == 0

    def test_descending_order(self) -> None:
        expr = np.array([0.0, 5.0, 1.0, 10.0, 3.0])
        ranked = rank_value_encode(expr)
        assert ranked[0] == 3  # gene 3 has highest expression (10.0)
        assert ranked[1] == 1  # gene 1 has second highest (5.0)

    def test_zeros_excluded(self) -> None:
        expr = np.array([0.0, 1.0, 0.0, 2.0])
        ranked = rank_value_encode(expr)
        assert 0 not in ranked
        assert 2 not in ranked
        assert len(ranked) == 2

    def test_global_median_weighting(self) -> None:
        """Rare genes (low median) get upweighted."""
        expr = np.array([2.0, 2.0, 2.0])
        medians = np.array([10.0, 0.1, 1.0])
        ranked = rank_value_encode(expr, medians)
        # Gene 1 has lowest median → highest weight → first
        assert ranked[0] == 1

    def test_single_gene(self) -> None:
        expr = np.array([0.0, 0.0, 5.0])
        ranked = rank_value_encode(expr)
        assert len(ranked) == 1
        assert ranked[0] == 2
