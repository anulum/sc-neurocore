# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEncoding from former test_datasets.py

"""Focused suite: TestEncoding from former test_datasets.py."""

from __future__ import annotations

from tests.datasets_support import *  # noqa: F403


class TestEncoding:
    def test_poisson_encode_shape(self):
        rates = np.array([0.5, 0.8, 0.1])
        spikes = poisson_encode(rates, T=100, seed=42)
        assert spikes.shape == (100, 3)
        assert spikes.dtype == np.bool_

    def test_poisson_encode_rate_correlation(self):
        rates = np.array([0.1, 0.9])
        spikes = poisson_encode(rates, T=10000, seed=0)
        empirical = spikes.mean(axis=0)
        np.testing.assert_allclose(empirical, rates, atol=0.05)

    def test_latency_encode_shape(self):
        values = np.array([0.2, 0.9, 0.5])
        spikes = latency_encode(values, T=50)
        assert spikes.shape == (50, 3)
        assert spikes.dtype == np.bool_

    def test_latency_encode_higher_value_fires_earlier(self):
        values = np.array([0.1, 0.9])
        spikes = latency_encode(values, T=100, tau=5.0)
        first_spike_0 = np.argmax(spikes[:, 0]) if spikes[:, 0].any() else 100
        first_spike_1 = np.argmax(spikes[:, 1]) if spikes[:, 1].any() else 100
        assert first_spike_1 <= first_spike_0

    def test_poisson_encode_zeros(self):
        spikes = poisson_encode(np.array([0.0]), T=100, seed=0)
        assert spikes.sum() == 0

    def test_poisson_encode_ones(self):
        spikes = poisson_encode(np.array([1.0]), T=100, seed=0)
        assert spikes.sum() == 100
