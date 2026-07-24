# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLatencyEncode from former test_encoding_zoo.py

"""Focused suite: TestLatencyEncode from former test_encoding_zoo.py."""

from __future__ import annotations

from tests.encoding_zoo_support import *  # noqa: F403


class TestLatencyEncode:
    def test_shape(self):
        s = latency_encode(np.array([0.5, 0.8, 0.2]), T=10)
        assert s.shape == (10, 3)

    def test_higher_value_fires_earlier(self):
        s = latency_encode(np.array([0.2, 0.9]), T=20)
        t_low = np.argmax(s[:, 0])
        t_high = np.argmax(s[:, 1])
        assert t_high <= t_low
