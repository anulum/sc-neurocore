# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrainScaleSIsolation from former test_model_brainscales_adex.py

"""Focused suite: TestBrainScaleSIsolation from former test_model_brainscales_adex.py."""

from __future__ import annotations

from tests.model_brainscales_adex_support import *  # noqa: F403


class TestBrainScaleSIsolation:
    def test_construction(self):
        n = BrainScaleSAdExNeuron()
        assert n.v == -65.0
        assert n.hw_speedup == 1000.0

    def test_step_returns_binary(self):
        n = BrainScaleSAdExNeuron()
        assert n.step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = BrainScaleSAdExNeuron()
        spikes = sum(n.step(5.0) for _ in range(10_000))
        assert spikes == 0, f"unexpected spikes at I=5: {spikes}"

    def test_spikes_under_drive(self):
        n = BrainScaleSAdExNeuron()
        spikes = sum(n.step(20.0) for _ in range(10_000))
        assert spikes > 0, "no spikes at I=20"

    def test_adaptation_variable(self):
        """w should increase after spiking (b=7 increment)."""
        n = BrainScaleSAdExNeuron()
        w_init = n.w
        for _ in range(10_000):
            n.step(20.0)
        assert n.w > w_init

    def test_exp_clipped(self):
        """Exponential term should not overflow (clipped to [-20, 20])."""
        n = BrainScaleSAdExNeuron()
        n.v = 1000.0  # extreme voltage
        result = n.step(0.0)
        assert np.isfinite(n.v)

    def test_state_finite(self):
        n = BrainScaleSAdExNeuron()
        for _ in range(10_000):
            n.step(30.0)
        assert np.isfinite(n.v)
        assert np.isfinite(n.w)

    def test_reset(self):
        n = BrainScaleSAdExNeuron()
        for _ in range(1000):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.w == 0.0
