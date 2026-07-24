# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikingKalmanFilter from former test_control.py

"""Focused suite: TestSpikingKalmanFilter from former test_control.py."""

from __future__ import annotations

from tests.control_support import *  # noqa: F403


class TestSpikingKalmanFilter:
    def test_step(self):
        kf = SpikingKalmanFilter(n_states=2, n_measurements=2)
        assert kf.step(np.array([1.0, 0.5])).shape == (2,)

    def test_reset(self):
        kf = SpikingKalmanFilter(2, 2)
        kf.step(np.ones(2))
        kf.reset()
        assert np.allclose(kf.x, 0)
