# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRMEdgeCases from former test_model_spike_response.py

"""Focused suite: TestSRMEdgeCases from former test_model_spike_response.py."""

from __future__ import annotations

from tests.model_spike_response_support import *  # noqa: F403


class TestSRMEdgeCases:
    def test_zero_input_silent(self):
        n = SpikeResponseNeuron()
        assert all(n.step(0.0) == 0 for _ in range(1000))

    def test_negative_input(self):
        n = SpikeResponseNeuron()
        n.time_since_spike = 1000.0
        n.step(-10.0)
        assert n.v < 0

    def test_time_since_spike_increments(self):
        n = SpikeResponseNeuron()
        tss0 = n.time_since_spike
        n.step(0.0)
        assert n.time_since_spike == tss0 + n.dt

    def test_spike_resets_tss_to_zero(self):
        n = SpikeResponseNeuron()
        n.step(10.0)
        assert n.time_since_spike == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SpikeResponseNeuron()
            trace = [(n.step(10.0), n.v, n.time_since_spike) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]
