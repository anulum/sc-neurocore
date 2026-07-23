# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTrainPCA from former test_spike_stats_dimensionality.py

"""Focused suite: TestSpikeTrainPCA from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403

class TestSpikeTrainPCA:
    def test_typical(self) -> None:
        proj, expl = spike_train_pca(_trains(), n_components=3)
        assert proj.shape == (3, 40)
        assert expl.shape == (3,)
        assert expl[0] >= expl[1] >= expl[2]
        assert expl.sum() <= 1.0 + 1e-9

    def test_python_backend(self) -> None:
        proj, expl = spike_train_pca(_trains(), n_components=2, backend="python")
        assert proj.shape == (2, 40)

    def test_empty_trains(self) -> None:
        proj, expl = spike_train_pca([])
        assert proj.size == 0 and expl.size == 0

    def test_single_neuron(self) -> None:
        proj, expl = spike_train_pca([np.tile([1, 0], 10).astype(np.int8)], bin_size=2)
        assert proj.shape[0] == 1
        npt.assert_array_equal(expl, [1.0])

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            spike_train_pca(_trains(), backend="cuda")
