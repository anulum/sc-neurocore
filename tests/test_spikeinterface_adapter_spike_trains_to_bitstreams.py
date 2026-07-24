# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTrainsToBitstreams from former test_spikeinterface_adapter.py

"""Focused suite: TestSpikeTrainsToBitstreams from former test_spikeinterface_adapter.py."""

from __future__ import annotations

from tests.spikeinterface_adapter_support import *  # noqa: F403


class TestSpikeTrainsToBitstreams:
    def test_output_shape(self):
        spikes = {0: np.array([10.0, 20.0]), 1: np.array([15.0])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=50.0, dt=1.0)
        assert mat.shape == (2, 50)

    def test_binary_output(self):
        spikes = {0: np.array([5.0, 10.0])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=20.0, dt=1.0)
        assert set(np.unique(mat)).issubset({0, 1})

    def test_spike_at_correct_bin(self):
        spikes = {0: np.array([5.0])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=10.0, dt=1.0)
        assert mat[0, 5] == 1
        assert mat[0, 0] == 0

    def test_empty_unit(self):
        spikes = {0: np.array([]), 1: np.array([3.0])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=10.0, dt=1.0)
        assert mat[0].sum() == 0
        assert mat[1].sum() == 1

    def test_dt_binning(self):
        spikes = {0: np.array([0.5, 1.5, 2.5])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=5.0, dt=2.0)
        assert mat.shape == (1, 3)  # ceil(5/2) = 3 bins
