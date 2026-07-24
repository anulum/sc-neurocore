# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRankOrderEncode from former test_encoding_zoo.py

"""Focused suite: TestRankOrderEncode from former test_encoding_zoo.py."""

from __future__ import annotations

from tests.encoding_zoo_support import *  # noqa: F403


class TestRankOrderEncode:
    def test_shape(self):
        s = rank_order_encode(np.array([0.3, 0.9, 0.1, 0.7]), T=10)
        assert s.shape == (10, 4)

    def test_order(self):
        s = rank_order_encode(np.array([0.1, 0.9, 0.5]), T=10)
        t_high = np.argmax(s[:, 1])
        t_low = np.argmax(s[:, 0])
        assert t_high <= t_low
