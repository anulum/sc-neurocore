# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSHDRealLoader from former test_datasets.py

"""Focused suite: TestSHDRealLoader from former test_datasets.py."""

from __future__ import annotations

from tests.datasets_support import *  # noqa: F403


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
        (tmp_path / "sentinel").touch()
        with pytest.raises(FileNotFoundError, match="not found"):
            load_shd(root=tmp_path, train=True, synthetic=False)


def test_load_shd_drops_spikes_after_the_window_instead_of_piling_them_up(tmp_path):
    """A real HDF5 file in the SHD layout: times in seconds, one ragged array per sample."""
    import h5py

    with h5py.File(tmp_path / "shd_train.h5", "w") as handle:
        times = handle.create_dataset("spikes/times", (1,), dtype=h5py.vlen_dtype(np.float32))
        units = handle.create_dataset("spikes/units", (1,), dtype=h5py.vlen_dtype(np.int32))
        # 1 ms bins, a 5-bin window: spikes at 1 ms and 3 ms fall inside, 9 ms and 20 ms do not.
        times[0] = np.array([0.001, 0.003, 0.009, 0.020], dtype=np.float32)
        units[0] = np.array([10, 11, 12, 13], dtype=np.int32)
        handle.create_dataset("labels", data=np.array([4]))

    samples, labels = load_shd(root=tmp_path, train=True, dt_ms=1.0, T=5)
    (sample,) = samples
    assert labels.tolist() == [4]
    assert sample.shape == (5, 700)
    assert sorted(zip(*np.nonzero(sample))) == [(1, 10), (3, 11)]
