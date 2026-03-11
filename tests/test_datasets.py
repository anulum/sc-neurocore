# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sc_neurocore.datasets.loaders import (
    _check_root,
    load_nmnist,
    load_shd,
    load_dvs_cifar10,
)
from sc_neurocore.datasets.encoding import poisson_encode, latency_encode


class TestCheckRoot:
    def test_valid_root_returns_path(self, tmp_path):
        p = _check_root(tmp_path, "test", "http://test")
        assert p == tmp_path

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="download from"):
            _check_root("/nonexistent_xyz_abc_123", "TestDS", "http://test")


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


class TestNMNISTRealLoader:
    """Test N-MNIST real-data path with synthetic .bin files."""

    def test_parse_nmnist_bin(self, tmp_path):
        from sc_neurocore.datasets.loaders import _parse_nmnist_bin

        rng = np.random.default_rng(0)
        n_events = 50
        raw = rng.integers(0, 256, size=(n_events, 5), dtype=np.uint8)
        bin_file = tmp_path / "sample.bin"
        raw.tofile(bin_file)
        events = _parse_nmnist_bin(bin_file, dt_ms=1.0)
        assert events.shape == (n_events, 4)
        assert events.dtype == np.float32

    def test_load_nmnist_real_path(self, tmp_path):
        rng = np.random.default_rng(0)
        split = tmp_path / "Train"
        for cls in range(3):
            d = split / str(cls)
            d.mkdir(parents=True)
            for s in range(2):
                raw = rng.integers(0, 256, size=(20, 5), dtype=np.uint8)
                (d / f"s{s}.bin").write_bytes(raw.tobytes())
        (split / "README.txt").touch()
        samples, labels = load_nmnist(root=tmp_path, train=True, synthetic=False)
        assert len(samples) == 6
        assert set(labels.tolist()) == {0, 1, 2}

    def test_load_nmnist_missing_split_raises(self, tmp_path):
        (tmp_path / "placeholder").touch()
        with pytest.raises(FileNotFoundError, match="Expected split directory"):
            load_nmnist(root=tmp_path, train=True, synthetic=False)


class TestSHDRealLoader:
    """Test SHD real-data path with mocked h5py."""

    def test_load_shd_real_path(self, tmp_path):
        rng = np.random.default_rng(0)
        n_samples = 5
        labs = rng.integers(0, 20, size=n_samples)

        spike_times_data = []
        spike_units_data = []
        for i in range(n_samples - 1):
            n_ev = int(rng.integers(10, 50))
            spike_times_data.append(rng.uniform(0.0, 1.0, size=n_ev).astype(np.float32))
            spike_units_data.append(rng.integers(0, 700, size=n_ev).astype(np.int32))
        spike_times_data.append(np.array([], dtype=np.float32))
        spike_units_data.append(np.array([], dtype=np.int32))

        h5_path = tmp_path / "shd_train.h5"
        h5_path.touch()

        class FakeDataset:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                if isinstance(key, slice):
                    return np.array(self._data)[key]
                return self._data[key]

            def __len__(self):
                return len(self._data)

        class FakeFile:
            def __init__(self):
                self._groups = {
                    "spikes": {
                        "times": FakeDataset(spike_times_data),
                        "units": FakeDataset(spike_units_data),
                    },
                    "labels": FakeDataset(labs),
                }

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def __getitem__(self, key):
                return self._groups[key]

        mock_h5py = MagicMock()
        mock_h5py.File.return_value = FakeFile()

        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            samples, labels_out = load_shd(root=tmp_path, train=True, synthetic=False)
        assert len(samples) == n_samples
        assert labels_out.shape == (n_samples,)
        assert samples[0].shape[1] == 700

    def test_load_shd_missing_h5_raises(self, tmp_path):
        (tmp_path / "placeholder").touch()
        with pytest.raises(FileNotFoundError, match="not found"):
            load_shd(root=tmp_path, train=True, synthetic=False)


class TestDVSCIFAR10RealLoader:
    """Test DVS-CIFAR10 real-data path with mock .npy files."""

    def test_load_dvs_cifar10_real_path(self, tmp_path):
        rng = np.random.default_rng(0)
        split = tmp_path / "train"
        for cls in range(3):
            d = split / str(cls)
            d.mkdir(parents=True)
            for s in range(2):
                events = rng.uniform(0, 128, size=(30, 4)).astype(np.float32)
                np.save(d / f"ev{s}.npy", events)
        (split / "README.txt").touch()
        samples, labels = load_dvs_cifar10(root=tmp_path, train=True, synthetic=False)
        assert len(samples) == 6
        assert set(labels.tolist()) == {0, 1, 2}
        assert samples[0].dtype == np.float32

    def test_load_dvs_cifar10_missing_split_raises(self, tmp_path):
        (tmp_path / "placeholder").touch()
        with pytest.raises(FileNotFoundError, match="Expected split directory"):
            load_dvs_cifar10(root=tmp_path, train=True, synthetic=False)

    def test_load_dvs_cifar10_empty_dir_raises(self, tmp_path):
        split = tmp_path / "train"
        split.mkdir()
        with pytest.raises(FileNotFoundError, match="No .npy event files"):
            load_dvs_cifar10(root=tmp_path, train=True, synthetic=False)
