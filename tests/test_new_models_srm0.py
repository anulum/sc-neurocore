# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRM0 from former test_new_models.py

"""Focused suite: TestSRM0 from former test_new_models.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.srm0 import (
    SRM0Neuron,
)


class TestSRM0:
    """The SRM0 spike-response neuron."""

    def test_fires_with_current(self) -> None:
        """A suprathreshold current makes the neuron fire."""
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, dt=0.1)
        spikes = sum(n.step(2.0) for _ in range(1000))
        assert spikes > 0

    def test_subthreshold_no_spikes(self) -> None:
        """A subthreshold current gives no spikes."""
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, dt=0.1)
        spikes = sum(n.step(0.5) for _ in range(1000))
        assert spikes == 0

    def test_eta_refractory(self) -> None:
        """The refractory kernel eta holds the voltage down after a spike."""
        n = SRM0Neuron(tau_m=20.0, v_threshold=1.0, eta_reset=10.0, dt=0.1)
        # Drive hard until spike
        for _ in range(1000):
            if n.step(2.0):
                break
        # Right after spike, v should be near rest due to eta
        assert n.v < 0.5

    def test_reset(self) -> None:
        """A reset returns v to v_rest and clears eta."""
        n = SRM0Neuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == n.v_rest
        assert n._eta == 0.0

    def test_get_state(self) -> None:
        """The state dictionary holds v, eta and t."""
        n = SRM0Neuron()
        n.step(0.1)
        s = n.get_state()
        assert "v" in s and "eta" in s and "t" in s

    def test_rate_increases_with_current(self) -> None:
        """A larger current gives a higher firing rate."""
        n1 = SRM0Neuron(dt=0.1)
        n2 = SRM0Neuron(dt=0.1)
        r1 = sum(n1.step(8.0) for _ in range(5000))
        r2 = sum(n2.step(15.0) for _ in range(5000))
        assert r2 > r1
