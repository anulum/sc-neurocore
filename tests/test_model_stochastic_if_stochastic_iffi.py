# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticIFFI from former test_model_stochastic_if.py

"""Focused suite: TestStochasticIFFI from former test_model_stochastic_if.py."""

from __future__ import annotations

from tests.model_stochastic_if_support import *  # noqa: F403


class TestStochasticIFFI:
    def test_subthreshold_deterministic_silent(self):
        """I=10 with sigma=0 → no spikes (V_ss = V_rest + I = -60 < -50)."""
        n = StochasticIFNeuron(sigma=0.0)
        assert len(_run(n, current=10.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = StochasticIFNeuron()
        assert len(_run(n, current=25.0, steps=10000)) >= 50

    def test_rate_increases_with_current(self):
        n20 = StochasticIFNeuron()
        n50 = StochasticIFNeuron()
        s20 = len(_run(n20, current=20.0, steps=50000))
        s50 = len(_run(n50, current=50.0, steps=50000))
        assert s50 > s20

    def test_rate_increases_with_sigma(self):
        """More noise → more noise-driven spikes → higher rate (near threshold)."""
        n_low = StochasticIFNeuron(sigma=1.0)
        n_high = StochasticIFNeuron(sigma=10.0)
        s_low = len(_run(n_low, current=18.0, steps=50000))
        s_high = len(_run(n_high, current=18.0, steps=50000))
        assert s_high > s_low
