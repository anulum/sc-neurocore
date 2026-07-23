# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPSNScoring from former test_model_psn.py

"""Focused suite: TestPSNScoring from former test_model_psn.py."""

from __future__ import annotations

from tests.model_psn_support import *  # noqa: F403

class TestPSNScoring:
    def test_spike_at_threshold(self):
        """With uniform kernel: score = mean(buffer). Spike when mean >= θ."""
        n = ParallelSpikingNeuron(kernel_size=4, v_threshold=1.0)
        # Fill buffer with 1.0 → score = mean([1,1,1,1]) = 1.0 → spike
        for i in range(3):
            n.step(1.0)
        s = n.step(1.0)  # 4th step fills buffer → score = 1.0 → spike
        assert s == 1

    def test_subthreshold_no_spike(self):
        """Score below threshold → no spike."""
        n = ParallelSpikingNeuron(kernel_size=8, v_threshold=1.0)
        # I=0.5 → avg=0.5 < 1.0
        spikes = sum(n.step(0.5) for _ in range(100))
        assert spikes == 0

    def test_buffer_cleared_on_spike(self):
        """After spike, buffer is zeroed → next step starts fresh."""
        n = ParallelSpikingNeuron(kernel_size=4, v_threshold=1.0)
        # Fill and trigger spike
        for _ in range(4):
            n.step(2.0)
        # Buffer should now be zeros
        assert np.all(n.buffer == 0.0)

    def test_rate_proportional_to_input(self):
        """At I=threshold: spikes every kernel_size steps (refill cycle)."""
        n = ParallelSpikingNeuron(kernel_size=8, v_threshold=1.0)
        spikes = sum(n.step(1.0) for _ in range(500))
        # Spikes every 8 steps (fill buffer, spike, clear, repeat)
        expected = 500 // 8
        assert abs(spikes - expected) <= 2

    def test_double_input_double_rate(self):
        """I=2*θ → score reaches threshold with half the buffer filled."""
        n1 = ParallelSpikingNeuron(kernel_size=8, v_threshold=1.0)
        n2 = ParallelSpikingNeuron(kernel_size=8, v_threshold=1.0)
        s1 = sum(n1.step(1.0) for _ in range(500))
        s2 = sum(n2.step(2.0) for _ in range(500))
        assert s2 > s1
