# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.datasets.loaders import load_nmnist, load_shd, load_dvs_cifar10
from sc_neurocore.datasets.encoding import poisson_encode, latency_encode


class TestSyntheticLoaders:
    def test_nmnist_synthetic_shapes(self):
        samples, labels = load_nmnist(synthetic=True, n_samples=10, seed=42)
        assert len(samples) == 10
        assert labels.shape == (10,)
        assert all(0 <= l < 10 for l in labels)

    def test_shd_synthetic_shapes(self):
        samples, labels = load_shd(synthetic=True, n_samples=5, seed=0)
        assert len(samples) == 5
        assert labels.shape == (5,)
        assert all(0 <= l < 20 for l in labels)

    def test_dvs_cifar10_synthetic_shapes(self):
        samples, labels = load_dvs_cifar10(synthetic=True, n_samples=8, seed=1)
        assert len(samples) == 8
        assert labels.shape == (8,)

    def test_nmnist_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            load_nmnist(root="/nonexistent/path", synthetic=False)

    def test_shd_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            load_shd(root="/nonexistent/path", synthetic=False)

    def test_dvs_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dvs_cifar10(root="/nonexistent/path", synthetic=False)

    def test_reproducibility(self):
        s1, l1 = load_nmnist(synthetic=True, n_samples=5, seed=99)
        s2, l2 = load_nmnist(synthetic=True, n_samples=5, seed=99)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(s1[0], s2[0])


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
