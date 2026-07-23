# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRM0 from former test_new_models.py

"""Focused suite: TestSRM0 from former test_new_models.py."""

from __future__ import annotations

from tests.new_models_support import *  # noqa: F403

class TestSRM0:
    def test_fires_with_current(self):
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, dt=0.1)
        spikes = sum(n.step(2.0) for _ in range(1000))
        assert spikes > 0

    def test_subthreshold_no_spikes(self):
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, dt=0.1)
        spikes = sum(n.step(0.5) for _ in range(1000))
        assert spikes == 0

    def test_eta_refractory(self):
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, eta_reset=10.0, dt=0.1)
        # Drive hard until spike
        for _ in range(1000):
            if n.step(2.0):
                break
        # Right after spike, v should be near rest due to eta
        assert n.v < 0.5

    def test_reset(self):
        n = SRM0Neuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == n.v_rest
        assert n._eta == 0.0

    def test_get_state(self):
        n = SRM0Neuron()
        n.step(0.1)
        s = n.get_state()
        assert "v" in s and "eta" in s and "t" in s

    def test_rate_increases_with_current(self):
        n1 = SRM0Neuron(dt=0.1)
        n2 = SRM0Neuron(dt=0.1)
        r1 = sum(n1.step(8.0) for _ in range(5000))
        r2 = sum(n2.step(15.0) for _ in range(5000))
        assert r2 > r1
