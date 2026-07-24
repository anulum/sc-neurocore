# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhaseEncode from former test_encoding_zoo.py

"""Focused suite: TestPhaseEncode from former test_encoding_zoo.py."""

from __future__ import annotations

from tests.encoding_zoo_support import *  # noqa: F403


class TestPhaseEncode:
    def test_shape(self):
        s = phase_encode(np.array([0.5, 0.3]), T=16, n_phases=4)
        assert s.shape == (16, 2)

    def test_periodic(self):
        s = phase_encode(np.array([0.5]), T=24, n_phases=8)
        assert s[:, 0].sum() >= 3  # fires every 8 steps at phase 4
