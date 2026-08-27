# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPSNIsolation from former test_model_psn.py

"""Focused suite: TestPSNIsolation from former test_model_psn.py."""

from __future__ import annotations

from tests.model_psn_support import *  # noqa: F403


class TestPSNIsolation:
    def test_construction_defaults(self):
        n = SCResettingParallelSpikingNeuron()
        assert n.kernel_size == 8
        assert n.v_threshold == 1.0
        assert n.kernel.shape == (8,)
        assert n.buffer.shape == (8,)

    def test_step_returns_binary(self):
        assert SCResettingParallelSpikingNeuron().step(0.0) in (0, 1)

    def test_default_kernel_is_uniform(self):
        """Default kernel = 1/kernel_size (averaging filter)."""
        n = SCResettingParallelSpikingNeuron(kernel_size=4)
        np.testing.assert_allclose(n.kernel, [0.25, 0.25, 0.25, 0.25])

    def test_buffer_fills_circularly(self):
        """Input values are written to circular buffer."""
        n = SCResettingParallelSpikingNeuron(kernel_size=4, v_threshold=100.0)
        for i in range(6):
            n.step(float(i))
        # Buffer wraps: positions 0,1,2,3 → values 4,5,2,3
        assert n.buffer[0] == 4.0
        assert n.buffer[1] == 5.0

    def test_reset(self):
        n = SCResettingParallelSpikingNeuron()
        for _ in range(20):
            n.step(5.0)
        n.reset()
        assert np.all(n.buffer == 0.0)
        assert n._ptr == 0
