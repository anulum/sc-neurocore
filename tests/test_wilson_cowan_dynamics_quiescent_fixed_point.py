# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuiescentFixedPoint from former test_wilson_cowan_dynamics.py

"""Focused suite: TestQuiescentFixedPoint from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403

class TestQuiescentFixedPoint:
    def test_zero_drive_converges_low(self):
        u = WilsonCowanUnit()
        for _ in range(10_000):
            u.step(0.0)
        assert u.e < 0.01
        assert u.i < 0.01

    def test_small_drive_stays_low(self):
        """Below the sigmoid's activation threshold, E and I stay near 0."""
        u = WilsonCowanUnit()
        for _ in range(5_000):
            u.step(0.2)
        assert u.e < 0.1
        assert u.i < 0.1
