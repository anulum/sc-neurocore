# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuronMicroProperties from former test_brunel_regression.py

"""Focused suite: TestNeuronMicroProperties from former test_brunel_regression.py."""

from __future__ import annotations

from tests.brunel_regression_support import *  # noqa: F403

class TestNeuronMicroProperties:
    """Fast single-neuron tests verifying biophysical properties."""

    def test_lif_subthreshold_decay(self):
        """Below threshold, membrane voltage decays toward v_rest."""
        bp = BrunelParams(v_threshold=20.0, v_reset=0.0, v_rest=0.0)
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        n.v = 15.0
        for _ in range(100):
            n.step(0.0)
        assert n.v < 15.0, "Membrane must decay without input"
        assert n.v >= 0.0, "Membrane must not go below v_rest"

    def test_lif_reset_value(self):
        """After spiking, membrane resets to v_reset."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        n.v = 25.0
        spike = n.step(0.0)
        assert spike == 1
        assert n.v == bp.v_reset

    def test_exact_vs_euler_leak_difference(self):
        """Exact exponential leak and Euler leak produce slightly different voltages."""
        bp = BrunelParams(v_threshold=20.0, v_rest=0.0, dt=0.1, tau_mem=20.0)
        p1 = translate_v1_stochastic_lif(bp)
        p10 = translate_v10_exact_leak(bp)

        n_euler = StochasticLIFNeuron(**p1["neuron_kwargs"])
        n_exact = StochasticLIFNeuron(**p10["neuron_kwargs"])

        n_euler.v = 15.0
        n_exact.v = 15.0

        # Euler: v += dt/tau * (v_rest - v) = 0.1/20 * (0 - 15) = -0.075 → v = 14.925
        n_euler.step(0.0)
        # Exact: v *= exp(-0.1/20) = 0.99501... → v = 14.925...
        n_exact.v = 15.0 * p10["leak_factor"]

        # Both should be close but not identical
        assert abs(n_euler.v - n_exact.v) < 0.01
        assert n_euler.v != n_exact.v  # Euler has truncation error

    def test_vectorized_params_completeness(self):
        """V20 translator must provide all fields needed for batch numpy update."""
        bp = BrunelParams()
        p = translate_v20_vectorized_numpy(bp)
        required = {
            "v_threshold",
            "v_reset",
            "v_rest",
            "tau_mem",
            "dt",
            "weight_exc",
            "weight_inh",
            "n_total",
            "n_exc",
        }
        missing = required - set(p.keys())
        assert not missing, f"V20 missing keys: {missing}"
