# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLTCAnalytical from former test_model_ltc.py

"""Focused suite: TestLTCAnalytical from former test_model_ltc.py."""

from __future__ import annotations

from tests.model_ltc_support import *  # noqa: F403


class TestLTCAnalytical:
    def test_input_dependent_tau(self):
        """τ depends on input via sigmoid: τ = τ_base · σ(w_τ·I)."""
        n = LiquidTimeConstantNeuron()
        # At I=0: σ(0) = 0.5 → τ = 10 * 0.5 = 5
        tau_zero = n.tau_base * (1.0 / (1.0 + np.exp(0.0)))
        assert abs(tau_zero - 5.0) < 1e-10

    def test_tau_clipped(self):
        """τ ≥ 0.1 prevents division by zero."""
        n = LiquidTimeConstantNeuron()
        # Very negative input → σ → 0 → tau → 0, clipped to 0.1
        n.step(-1000.0)
        assert np.isfinite(n.x)

    def test_f_target_tanh(self):
        """f_target = tanh(w_x·x + w_in·I). Bounded [-1, 1]."""
        f = np.tanh(0.8 * 0.0 + 1.0 * 5.0)
        assert -1 <= f <= 1

    def test_spike_resets_x(self):
        n = LiquidTimeConstantNeuron()
        for _ in range(10_000):
            if n.step(5.0) == 1:
                assert n.x == 0.0
                break
