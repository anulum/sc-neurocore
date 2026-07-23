# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeltaEncode from former test_encoding_zoo.py

"""Focused suite: TestDeltaEncode from former test_encoding_zoo.py."""

from __future__ import annotations

from tests.encoding_zoo_support import *  # noqa: F403

class TestDeltaEncode:
    def test_1d(self):
        signal = np.array([0.0, 0.0, 0.5, 0.5, 1.0])
        s = delta_encode(signal, threshold=0.3)
        assert s.shape == (5, 1)
        assert s[2, 0] == 1  # change 0→0.5

    def test_2d(self):
        signal = np.random.rand(20, 4)
        s = delta_encode(signal, threshold=0.1)
        assert s.shape == (20, 4)
