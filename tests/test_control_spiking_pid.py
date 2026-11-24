# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikingPID from former test_control.py

"""Focused suite: TestSpikingPID from former test_control.py."""

from __future__ import annotations

from sc_neurocore.control import (
    SpikingPID,
)


class TestSpikingPID:
    """The spiking PID controller."""

    def test_step(self) -> None:
        """An error gives a non-zero output."""
        assert SpikingPID(Kp=1.0).step(1.0) != 0

    def test_converges(self) -> None:
        """The controlled state approaches its set point."""
        pid = SpikingPID(Kp=0.5, Ki=0.01, Kd=0.001, dt=0.1)
        state = 0.0
        for _ in range(200):
            state += pid.step(1.0 - state) * 0.1
        assert abs(state - 1.0) < 1.0

    def test_spike_output(self) -> None:
        """The spiking output has one entry per neuron and term."""
        assert SpikingPID(n_neurons=5).step_spike(0.5).shape == (15,)

    def test_reset(self) -> None:
        """A reset clears the integral."""
        pid = SpikingPID()
        pid.step(1.0)
        pid.reset()
        assert pid._integral == 0.0
