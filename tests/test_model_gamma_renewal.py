# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GammaRenewalNeuron

"""Full pipeline test for GammaRenewalNeuron (Keat et al. 2001).

ISI ~ Gamma(k, k/rate). Hazard-based stochastic spiking."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.gamma_renewal import (
    GammaRenewalNeuron,
    _log_gamma_int,
    _gamma_survival,
)
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestGammaRenewalIsolation:
    def test_construction(self):
        n = GammaRenewalNeuron()
        assert n.rate_hz == 50.0
        assert n.shape_k == 3

    def test_step_returns_binary(self):
        assert GammaRenewalNeuron().step() in (0, 1)

    def test_spikes_at_default_rate(self):
        """50 Hz default → ~500 spikes in 10K steps (dt=1ms)."""
        n = GammaRenewalNeuron()
        s = sum(n.step() for _ in range(10000))
        assert 200 < s < 1000

    def test_rate_proportional(self):
        n_low = GammaRenewalNeuron(rate_hz=10.0)
        n_high = GammaRenewalNeuron(rate_hz=100.0)
        s_low = sum(n_low.step() for _ in range(10000))
        s_high = sum(n_high.step() for _ in range(10000))
        assert s_high > s_low * 3

    def test_rate_override(self):
        """rate_override parameter should replace base rate."""
        n = GammaRenewalNeuron(rate_hz=10.0)
        s = sum(n.step(rate_override=200.0) for _ in range(5000))
        assert s > 500

    def test_shape_k_effect(self):
        """Higher k → more regular ISIs (less variable)."""
        isi_k1, isi_k5 = [], []
        for k, isi_list in [(1, isi_k1), (5, isi_k5)]:
            n = GammaRenewalNeuron(shape_k=k, rate_hz=50.0)
            last = 0
            for t in range(20000):
                if n.step():
                    if last > 0:
                        isi_list.append(t - last)
                    last = t
        if len(isi_k1) > 10 and len(isi_k5) > 10:
            cv_k1 = np.std(isi_k1) / np.mean(isi_k1)
            cv_k5 = np.std(isi_k5) / np.mean(isi_k5)
            assert cv_k5 < cv_k1

    def test_time_since_spike_resets(self):
        """_time_since_spike should reset to 0 after a spike."""
        n = GammaRenewalNeuron(rate_hz=200.0)
        for _ in range(10000):
            if n.step():
                assert n._time_since_spike == 0.0
                break

    def test_numerical_stability(self):
        for rate in [1.0, 50.0, 200.0, 1000.0]:
            n = GammaRenewalNeuron(rate_hz=rate)
            for _ in range(5000):
                n.step()
            assert np.isfinite(n._time_since_spike)

    def test_reset(self):
        n = GammaRenewalNeuron()
        for _ in range(500):
            n.step()
        n.reset()
        assert n._time_since_spike == 0.0

    def test_zero_rate_no_spikes(self):
        n = GammaRenewalNeuron(rate_hz=0.0)
        assert sum(n.step() for _ in range(1000)) == 0


class TestGammaRenewalHelpers:
    def test_log_gamma_int(self):
        assert _log_gamma_int(1) == 0.0
        assert abs(_log_gamma_int(4) - np.log(6.0)) < 1e-10

    def test_gamma_survival_at_zero(self):
        assert _gamma_survival(3, 0.0) == 1.0

    def test_gamma_survival_decreases(self):
        s1 = _gamma_survival(3, 1.0)
        s2 = _gamma_survival(3, 5.0)
        assert s2 < s1


class TestGammaRenewalNetwork:
    def test_population(self):
        assert Population(GammaRenewalNeuron, n=10, label="gr").n == 10


class TestGammaRenewalAnalysis:
    def test_spike_count(self):
        n = GammaRenewalNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step()
        assert spike_count(train) > 200
