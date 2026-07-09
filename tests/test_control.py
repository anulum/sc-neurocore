# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from __future__ import annotations
import numpy as np
from sc_neurocore.control import SpikingPID, SpikingKalmanFilter, SpikingLQR


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


class TestSpikingKalmanFilter:
    def test_step(self):
        kf = SpikingKalmanFilter(n_states=2, n_measurements=2)
        assert kf.step(np.array([1.0, 0.5])).shape == (2,)

    def test_reset(self):
        kf = SpikingKalmanFilter(2, 2)
        kf.step(np.ones(2))
        kf.reset()
        assert np.allclose(kf.x, 0)


class TestSpikingLQR:
    def test_control(self):
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.0], [0.1]])
        lqr = SpikingLQR(A, B)
        assert lqr.control(np.array([1.0, 0.5])).shape == (1,)

    def test_gain(self):
        lqr = SpikingLQR(np.eye(2), np.array([[1.0], [0.0]]))
        assert lqr.gain_matrix.shape == (1, 2)

    def test_stabilizes(self):
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.005], [0.1]])
        lqr = SpikingLQR(A, B)
        x = np.array([10.0, 5.0])
        for _ in range(200):
            x = A @ x + B @ lqr.control(x)
        assert np.linalg.norm(x) < 10.0
