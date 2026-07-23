# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTimedArray from former test_network_monitors_stimulus.py

"""Focused suite: TestTimedArray from former test_network_monitors_stimulus.py."""

from __future__ import annotations

from tests.network_monitors_stimulus_support import *  # noqa: F403

class TestTimedArray:
    def test_returns_value_at_step(self):
        ta = TimedArray([0.0, 1.0, 2.0, 3.0], dt=0.001)
        assert ta.get_current(0) == 0.0
        assert ta.get_current(2) == 2.0

    def test_clamps_past_end(self):
        ta = TimedArray([5.0, 10.0], dt=0.001)
        assert ta.get_current(100) == 10.0

    def test_accepts_numpy_array(self):
        arr = np.linspace(0, 1, 50)
        ta = TimedArray(arr, dt=0.001)
        np.testing.assert_allclose(ta.get_current(25), arr[25])

    def test_single_value(self):
        ta = TimedArray([42.0])
        assert ta.get_current(0) == 42.0
        assert ta.get_current(999) == 42.0
