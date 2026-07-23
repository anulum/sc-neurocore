# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRefractoryRateCeiling from former test_brunel_regression.py

"""Focused suite: TestRefractoryRateCeiling from former test_brunel_regression.py."""

from __future__ import annotations

from tests.brunel_regression_support import *  # noqa: F403

class TestRefractoryRateCeiling:
    def test_refractory_limits_rate(self):
        """With 5-step refractory at dt=0.1ms, max rate = 1/(5*0.1ms) = 2000 Hz."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v8_refractory(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        spikes = 0
        steps = 10000
        for _ in range(steps):
            n.v += 25.0
            spikes += n.step(0.0)
        rate_hz = spikes / (steps * bp.dt / 1000.0)
        max_theoretical = 1000.0 / (params["neuron_kwargs"]["refractory_period"] * bp.dt)
        assert rate_hz <= max_theoretical * 1.01
