# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for spike/state/rate monitor edge paths

"""Contracts for residual edges in the spike, state and rate monitors."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.network.monitor import RateMonitor, SpikeMonitor, StateMonitor


class _Population:
    """Minimal population stand-in exposing the attributes the monitors need."""

    n = 4
    label = "pop"

    def get_states(self) -> dict[str, np.ndarray[Any, Any]]:
        return {"v": np.arange(4, dtype=np.float64)}


def test_spike_monitor_direct_event_and_degenerate_paths() -> None:
    """SpikeMonitor records direct events and handles zero duration and empty correlation."""
    monitor = SpikeMonitor(_Population())

    monitor.record_event(2, 5)
    assert monitor.count == 1

    np.testing.assert_array_equal(monitor.firing_rates(0), np.zeros(4))

    correlogram, lags = monitor.cross_correlation(0, 1)
    assert np.all(correlogram == 0)
    assert lags.size == correlogram.size

    for step in (1, 3, 5):
        monitor.record_event(0, step)
        monitor.record_event(1, step + 1)
    populated, populated_lags = monitor.cross_correlation(0, 1, max_lag=3)
    assert populated.size == populated_lags.size


def test_state_monitor_records_selected_subset() -> None:
    """StateMonitor with a record subset snapshots only the selected neurons."""
    monitor = StateMonitor(_Population(), variables=["v"], record=[0, 2])

    monitor.snapshot(0)

    assert monitor.traces["v"].shape[1] == 2


def test_rate_monitor_reports_empty_before_recording() -> None:
    """RateMonitor reports empty rate and bin-edge arrays before any recording."""
    monitor = RateMonitor(_Population())

    assert monitor.rate.size == 0
    assert monitor.t.size == 0
