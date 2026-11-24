# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExecutionTrace from former test_debug_toolkit.py

"""Focused suite: TestExecutionTrace from former test_debug_toolkit.py."""

from __future__ import annotations

import numpy as np
import pytest
from tests.debug_toolkit_support import (
    _make_trace,
)


class TestExecutionTrace:
    """The recorded execution trace."""

    def test_spike_count_zero(self) -> None:
        """A silent trace counts no spikes."""
        trace = _make_trace()
        assert trace.spike_count == 0

    def test_spike_count_nonzero(self) -> None:
        """Two spikes are counted."""
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[3, 1] = 1
        spikes[7, 4] = 1
        trace = _make_trace(spikes=spikes)
        assert trace.spike_count == 2

    def test_firing_rates(self) -> None:
        """Each neuron's firing rate is its spike share."""
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[0, 0] = 1
        spikes[5, 0] = 1
        trace = _make_trace(spikes=spikes)
        rates = trace.firing_rates
        assert rates[0] == pytest.approx(0.2)
        assert rates[1] == pytest.approx(0.0)

    def test_neuron_trace(self) -> None:
        """A neuron trace lists its spike times."""
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[2, 1] = 1
        spikes[7, 1] = 1
        voltages = np.random.randn(10, 5)
        currents = np.random.randn(10, 5)
        trace = _make_trace(spikes=spikes, voltages=voltages, currents=currents)
        nt = trace.neuron_trace(1)
        assert len(nt["spike_times"]) == 2
        assert nt["spike_times"][0] == 2
        assert nt["spike_times"][1] == 7

    def test_spike_times(self) -> None:
        """A neuron's spike times are listed."""
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[4, 3] = 1
        trace = _make_trace(spikes=spikes)
        times = trace.spike_times(3)
        assert len(times) == 1
        assert times[0] == 4

    def test_population_spikes(self) -> None:
        """Spikes are sliced by population."""
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[0, 0] = 1
        spikes[0, 4] = 1
        trace = _make_trace(spikes=spikes)
        pop_a = trace.population_spikes("pop_a")
        assert pop_a.shape == (10, 3)
        assert pop_a[0, 0] == 1
        pop_b = trace.population_spikes("pop_b")
        assert pop_b.shape == (10, 2)
        assert pop_b[0, 1] == 1

    def test_population_spikes_not_found(self) -> None:
        """An unknown population is rejected."""
        trace = _make_trace()
        with pytest.raises(ValueError, match="not found"):
            trace.population_spikes("nonexistent")
