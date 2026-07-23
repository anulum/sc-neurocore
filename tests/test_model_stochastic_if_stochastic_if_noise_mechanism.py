# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticIFNoiseMechanism from former test_model_stochastic_if.py

"""Focused suite: TestStochasticIFNoiseMechanism from former test_model_stochastic_if.py."""

from __future__ import annotations

from tests.model_stochastic_if_support import *  # noqa: F403

class TestStochasticIFNoiseMechanism:
    """Core: OU noise with amplitude sigma·sqrt(dt/tau_m)."""

    def test_sigma_zero_is_deterministic(self):
        """sigma=0 → identical runs (no RNG)."""
        np.random.seed(42)
        n1 = StochasticIFNeuron(sigma=0.0)
        t1 = [(n1.step(25.0), n1.v) for _ in range(200)]
        np.random.seed(42)
        n2 = StochasticIFNeuron(sigma=0.0)
        t2 = [(n2.step(25.0), n2.v) for _ in range(200)]
        assert t1 == t2

    def test_sigma_zero_constant_isi(self):
        """sigma=0: deterministic LIF → perfectly constant ISI."""
        n = StochasticIFNeuron(sigma=0.0)
        spikes = _run(n, current=25.0, steps=10000)
        if len(spikes) > 10:
            isis = np.diff(spikes[2:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.001, f"CV(ISI) = {cv:.4f} with sigma=0"

    def test_sigma_nonzero_variable_isi(self):
        """sigma>0: ISI has variability (CV > 0)."""
        n = StochasticIFNeuron(sigma=3.0)
        spikes = _run(n, current=25.0, steps=50000)
        assert len(spikes) >= 100
        isis = np.diff(spikes).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv > 0.05, f"CV(ISI) = {cv:.4f}, expected > 0.05 with noise"

    def test_noise_amplitude_scales_with_sigma(self):
        """Higher sigma → more ISI variability."""
        cv_low = _measure_cv(StochasticIFNeuron(sigma=1.0), 25.0, 50000)
        cv_high = _measure_cv(StochasticIFNeuron(sigma=5.0), 25.0, 50000)
        if cv_low is not None and cv_high is not None:
            assert cv_high > cv_low

    def test_noise_enables_subthreshold_spiking(self):
        """At I=15 (subthreshold for deterministic), noise (sigma=10) triggers spikes."""
        n_det = StochasticIFNeuron(sigma=0.0)
        n_noisy = StochasticIFNeuron(sigma=10.0)
        s_det = len(_run(n_det, current=15.0, steps=10000))
        s_noisy = len(_run(n_noisy, current=15.0, steps=10000))
        assert s_det == 0, "Deterministic should not spike at I=15"
        assert s_noisy > 10, "Noise should trigger spikes at subthreshold I"

    def test_two_runs_differ(self):
        """Two neurons with same params produce different spike trains."""
        n1 = StochasticIFNeuron()
        n2 = StochasticIFNeuron()
        t1 = [n1.step(25.0) for _ in range(1000)]
        t2 = [n2.step(25.0) for _ in range(1000)]
        assert t1 != t2
