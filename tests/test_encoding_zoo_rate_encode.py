# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRateEncode from former test_encoding_zoo.py

"""Focused suite: TestRateEncode from former test_encoding_zoo.py."""

from __future__ import annotations

from tests.encoding_zoo_support import *  # noqa: F403

class TestRateEncode:
    def test_shape(self):
        s = rate_encode(np.array([0.5, 0.3, 0.8]), T=20)
        assert s.shape == (20, 3)
        assert s.dtype == np.int8

    def test_rate_correlation(self):
        s = rate_encode(np.array([0.1, 0.9]), T=1000, seed=42)
        assert s[:, 1].mean() > s[:, 0].mean()
