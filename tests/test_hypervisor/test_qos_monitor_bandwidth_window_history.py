# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The bandwidth meter keeps only the samples its window can read

"""Hold the bandwidth meter's history to one window and its base, without changing any rate.

A tenant's cycles increase, so each later window's base is the current base or a later sample;
the samples before it can never be read again. The meter drops them when it records, and every
rate equals the one the whole history gives.
"""

from __future__ import annotations

import random

import pytest

from sc_neurocore.hypervisor.qos_monitor import BandwidthMeter


def _full_history_rate(counts: list[int], stamps: list[int], window_cycles: int) -> float:
    """Return the windowed rate over a never-pruned history, as the meter defines it."""
    if len(counts) < 2:
        return 0.0
    cutoff = stamps[-1] - window_cycles
    base = max((i for i, cycle in enumerate(stamps) if cycle <= cutoff), default=0)
    span = stamps[-1] - stamps[base]
    return sum(counts[base + 1 :]) / span if span > 0 else 0.0


@pytest.mark.parametrize("seed", range(20))
def test_every_rate_equals_the_rate_of_the_whole_history(seed: int) -> None:
    """After each record, the pruned meter's rate is the whole history's rate."""
    rng = random.Random(seed)
    window = rng.choice([1, 5, 50, 1_000, 100_000])
    meter = BandwidthMeter(window_cycles=window)
    counts: list[int] = []
    stamps: list[int] = []
    cycle = rng.randrange(0, 50)
    for _ in range(400):
        cycle += rng.randrange(1, 60)
        spikes = rng.randrange(0, 40)
        meter.record("t0", spikes, cycle)
        counts.append(spikes)
        stamps.append(cycle)
        assert meter.throughput("t0") == _full_history_rate(counts, stamps, window)


def test_a_long_run_keeps_the_history_within_one_window_and_its_base() -> None:
    """Fifty thousand samples seven cycles apart leave at most one window of them and the base."""
    window, step = 1_000, 7
    meter = BandwidthMeter(window_cycles=window)
    for n in range(1, 50_001):
        meter.record("t0", 3, n * step)
    stamps = meter._timestamps["t0"]
    assert len(meter._counters["t0"]) == len(stamps)
    assert len(stamps) <= window // step + 2
    cutoff = stamps[-1] - window
    assert stamps[0] <= cutoff < stamps[1]
    assert meter.throughput("t0") == pytest.approx(3 / step)


def test_tenants_are_pruned_apart() -> None:
    """One tenant's long run leaves another tenant's short history whole."""
    meter = BandwidthMeter(window_cycles=10)
    meter.record("quiet", 1, 1)
    meter.record("quiet", 1, 2)
    for cycle in range(1, 200):
        meter.record("busy", 1, cycle)
    assert meter._timestamps["quiet"] == [1, 2]
    assert len(meter._timestamps["busy"]) == 11


def test_a_refused_cycle_leaves_the_pruned_history_as_it_was() -> None:
    """A cycle that is not later than the last is refused before anything is dropped."""
    meter = BandwidthMeter(window_cycles=10)
    for cycle in range(0, 100, 5):
        meter.record("t0", 2, cycle)
    before = (list(meter._counters["t0"]), list(meter._timestamps["t0"]))
    with pytest.raises(ValueError, match="not later than"):
        meter.record("t0", 9, 95)
    assert (meter._counters["t0"], meter._timestamps["t0"]) == before
