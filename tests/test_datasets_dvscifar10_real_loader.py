# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDVSCIFAR10RealLoader from former test_datasets.py

"""Focused suite: TestDVSCIFAR10RealLoader from former test_datasets.py."""

from __future__ import annotations

from tests.datasets_support import *  # noqa: F403


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
        (tmp_path / "sentinel").touch()
        with pytest.raises(FileNotFoundError, match="Expected split directory"):
            load_dvs_cifar10(root=tmp_path, train=True, synthetic=False)

    def test_load_dvs_cifar10_empty_dir_raises(self, tmp_path):
        split = tmp_path / "train"
        split.mkdir()
        with pytest.raises(FileNotFoundError, match="No .npy event files"):
            load_dvs_cifar10(root=tmp_path, train=True, synthetic=False)
