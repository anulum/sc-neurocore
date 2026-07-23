# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBurstEncode from former test_encoding_zoo.py

"""Focused suite: TestBurstEncode from former test_encoding_zoo.py."""

from __future__ import annotations

from tests.encoding_zoo_support import *  # noqa: F403

class TestBurstEncode:
    def test_shape(self):
        s = burst_encode(np.array([0.2, 0.8]), T=10, max_burst=5)
        assert s.shape == (10, 2)

    def test_higher_value_longer_burst(self):
        s = burst_encode(np.array([0.2, 1.0]), T=10, max_burst=5)
        assert s[:, 1].sum() >= s[:, 0].sum()
