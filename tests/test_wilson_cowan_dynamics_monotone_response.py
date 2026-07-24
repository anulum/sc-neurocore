# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMonotoneResponse from former test_wilson_cowan_dynamics.py

"""Focused suite: TestMonotoneResponse from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403


class TestMonotoneResponse:
    def test_stronger_drive_higher_e(self):
        finals = []
        for drive in (0.5, 1.5, 3.0, 5.0, 8.0):
            u = WilsonCowanUnit()
            for _ in range(5_000):
                u.step(drive)
            finals.append(u.e)
        # E saturates near 1 but must be monotonically non-decreasing
        # with drive strength.
        diffs = np.diff(finals)
        assert (diffs >= -1e-3).all(), f"E should be non-decreasing with drive; got {finals}"
        assert finals[-1] > finals[0] + 0.3
