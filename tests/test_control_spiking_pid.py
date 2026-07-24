# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikingPID from former test_control.py

"""Focused suite: TestSpikingPID from former test_control.py."""

from __future__ import annotations

from tests.control_support import *  # noqa: F403


class TestSpikingPID:
    def test_step(self):
        assert SpikingPID(Kp=1.0).step(1.0) != 0

    def test_converges(self):
        pid = SpikingPID(Kp=0.5, Ki=0.01, Kd=0.001, dt=0.1)
        state = 0.0
        for _ in range(200):
            state += pid.step(1.0 - state) * 0.1
        assert abs(state - 1.0) < 1.0

    def test_spike_output(self):
        assert SpikingPID(n_neurons=5).step_spike(0.5).shape == (15,)

    def test_reset(self):
        pid = SpikingPID()
        pid.step(1.0)
        pid.reset()
        assert pid._integral == 0.0
