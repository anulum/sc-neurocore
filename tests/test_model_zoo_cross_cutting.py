# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossCutting from former test_model_zoo.py

"""Focused suite: TestCrossCutting from former test_model_zoo.py."""

from __future__ import annotations

from sc_neurocore.network.network import Network
from collections.abc import Callable
import numpy as np
import pytest
from sc_neurocore.analysis.spike_stats.basic import (
    isi,
    spike_count,
)
from tests.model_zoo_support import (
    _ALL_BUILDERS,
    _total_spikes,
)


class TestCrossCutting:
    """Properties that must hold for every model_zoo configuration."""

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_has_populations(self, name: str, builder: Callable[[], Network]) -> None:
        """Every configuration has at least two populations."""
        net = builder()
        assert len(net.populations) >= 2

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_has_projections(self, name: str, builder: Callable[[], Network]) -> None:
        """Every configuration has at least one projection."""
        net = builder()
        assert len(net.projections) >= 1

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_has_spike_monitors(self, name: str, builder: Callable[[], Network]) -> None:
        """Every configuration has at least one spike monitor."""
        net = builder()
        assert len(net.spike_monitors) >= 1

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_has_stimuli(self, name: str, builder: Callable[[], Network]) -> None:
        """Every configuration has at least one stimulus."""
        net = builder()
        assert len(net.stimuli) >= 1

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_seed_determinism(self, name: str, builder: Callable[[], Network]) -> None:
        """Same seed → same spike count."""
        net1 = builder()
        net1.run(0.05, dt=0.001, backend="python")
        c1 = _total_spikes(net1)
        net2 = builder()
        net2.run(0.05, dt=0.001, backend="python")
        c2 = _total_spikes(net2)
        assert c1 == c2

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_analysis_spike_count_all(self, name: str, builder: Callable[[], Network]) -> None:
        """spike_count works on monitor data from every config."""
        net = builder()
        net.run(0.1, dt=0.001, backend="python")
        for mon in net.spike_monitors:
            train = np.zeros(100, dtype=float)
            for t in mon.spike_times:
                if 0 <= t < 100:
                    train[t] = 1.0
            assert spike_count(train) >= 0

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_analysis_isi_all(self, name: str, builder: Callable[[], Network]) -> None:
        """ISI computation works on per-neuron binary trains from every config."""
        net = builder()
        n_steps = 100
        net.run(0.1, dt=0.001, backend="python")
        for mon in net.spike_monitors:
            trains = mon.spike_trains
            for nid, times in trains.items():
                if len(times) >= 3:
                    # Build binary train for this single neuron
                    binary = np.zeros(n_steps, dtype=float)
                    for t in times:
                        if 0 <= t < n_steps:
                            binary[t] = 1.0
                    intervals = isi(binary, dt=0.001)
                    if intervals.size > 0:
                        assert np.all(intervals > 0)
                    break  # one neuron sufficient per monitor
