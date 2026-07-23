# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaEncode from former test_encoding_zoo.py

"""Focused suite: TestSigmaDeltaEncode from former test_encoding_zoo.py."""

from __future__ import annotations

from tests.encoding_zoo_support import *  # noqa: F403

class TestSigmaDeltaEncode:
    def test_1d(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        s = sigma_delta_encode(signal, threshold=0.2)
        assert s.shape == (100, 1)
        assert s.sum() > 0

    def test_2d(self):
        signal = np.random.rand(50, 3)
        s = sigma_delta_encode(signal, threshold=0.1)
        assert s.shape == (50, 3)
