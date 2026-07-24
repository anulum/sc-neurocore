# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRMIsolation from former test_model_spike_response.py

"""Focused suite: TestSRMIsolation from former test_model_spike_response.py."""

from __future__ import annotations

from tests.model_spike_response_support import *  # noqa: F403


class TestSRMIsolation:
    def test_construction_defaults(self):
        n = SpikeResponseNeuron()
        assert n.v == 0.0
        assert n.v_threshold == 1.0
        assert n.tau_eta == 10.0
        assert n.tau_kappa == 5.0
        assert n.eta_reset == -5.0
        assert n.time_since_spike == 1000.0
        assert n.dt == 1.0

    def test_step_returns_binary(self):
        assert SpikeResponseNeuron().step(0.0) in (0, 1)

    def test_v_no_accumulation(self):
        """v = η + κ, computed fresh. No memory of previous v."""
        n = SpikeResponseNeuron()
        n.step(5.0)
        v1 = n.v
        n.step(5.0)
        v2 = n.v
        # v should be nearly identical (both have tss > 1000, eta ≈ 0)
        assert abs(v2 - v1) < 0.01

    def test_reset(self):
        n = SpikeResponseNeuron()
        for _ in range(50):
            n.step(10.0)
        n.reset()
        assert n.v == 0.0
        assert n.time_since_spike == 1000.0
