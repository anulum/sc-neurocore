# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikingLQR from former test_control.py

"""Focused suite: TestSpikingLQR from former test_control.py."""

from __future__ import annotations

from tests.control_support import *  # noqa: F403


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
