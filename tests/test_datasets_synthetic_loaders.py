# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSyntheticLoaders from former test_datasets.py

"""Focused suite: TestSyntheticLoaders from former test_datasets.py."""

from __future__ import annotations

from tests.datasets_support import *  # noqa: F403

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
