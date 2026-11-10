#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore — Switching activity comparison: clock-driven vs event-driven LIF
#
# Counts membrane-register bit toggles of two Python models, one of each
# neuron, under identical sparse input. Toggles are one input of dynamic
# power; clock distribution, the leak counter and control logic are not
# counted, so the printed figures are not a power measurement.

"""Compare membrane-register toggles of a clock-driven and an event-driven LIF model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_ACTIVITIES = (10.0, 1.0, 0.1, 0.01)
DEFAULT_CYCLES = 100_000


def _q88_encode(val: float) -> int:
    raw = int(round(val * 256))
    return raw & 0xFFFF


def _q88_sat_add(a: int, b: int) -> int:
    s = a + b
    if s > 32767:
        return 32767
    if s < -32768:
        return -32768
    return s


def _q88_sat_mul(a: int, b: int) -> int:
    product = a * b
    result = product >> 8
    if result > 32767:
        return 32767
    if result < -32768:
        return -32768
    return result


def _count_toggles(prev: int, curr: int, width: int = 16) -> int:
    xor = (prev ^ curr) & ((1 << width) - 1)
    return bin(xor).count("1")


def simulate_clock_driven(
    n_cycles: int, spike_times: set[int], leak_k: int, gain_k: int, threshold: int
) -> tuple[int, list[int]]:
    """Model sc_lif_neuron.v, which updates its membrane every cycle.

    Parameters
    ----------
    n_cycles : int
        Number of clock cycles to simulate.
    spike_times : set of int
        Cycles at which an input spike arrives.
    leak_k : int
        Leak constant in Q8.8.
    gain_k : int
        Input gain in Q8.8.
    threshold : int
        Firing threshold in Q8.8.

    Returns
    -------
    tuple of (int, list of int)
        Membrane-register bit toggles, and the cycles at which the neuron fired.
    """
    v = 0
    prev_v = 0
    total_toggles = 0
    spikes_out = []

    for t in range(n_cycles):
        I_t = _q88_encode(0.5) if t in spike_times else 0

        # LIF update (every cycle)
        dv_leak = _q88_sat_mul(0 - v, leak_k)  # (V_REST - v) * leak_k >> 8
        dv_in = _q88_sat_mul(I_t, gain_k)
        v_next = _q88_sat_add(v, _q88_sat_add(dv_leak, dv_in))

        if v_next >= threshold:
            spikes_out.append(t)
            v_next = 0  # reset

        total_toggles += _count_toggles(prev_v, v_next)
        prev_v = v_next
        v = v_next

    return total_toggles, spikes_out


def simulate_event_driven(
    n_cycles: int,
    spike_times: set[int],
    leak_k: int,
    threshold: int,
    leak_period: int,
) -> tuple[int, list[int]]:
    """Model sc_event_neuron.v, which integrates on input events and decays on leak ticks.

    Parameters
    ----------
    n_cycles : int
        Number of clock cycles to simulate.
    spike_times : set of int
        Cycles at which an input event arrives.
    leak_k : int
        Leak constant in Q8.8.
    threshold : int
        Firing threshold in Q8.8.
    leak_period : int
        Cycles between leak ticks (the RTL's ``LEAK_PERIOD`` parameter).

    Returns
    -------
    tuple of (int, list of int)
        Membrane-register bit toggles, and the cycles at which the neuron fired.
    """
    v = 0
    prev_v = 0
    total_toggles = 0
    spikes_out = []
    leak_counter = leak_period - 1

    for t in range(n_cycles):
        v_next = v

        # Leak: only on timer tick
        if leak_counter == 0:
            v_next = _q88_sat_mul(v_next, leak_k)
            leak_counter = leak_period - 1
        else:
            leak_counter -= 1

        # Event: only when spike arrives
        if t in spike_times:
            weight = _q88_encode(0.5)
            v_next = _q88_sat_add(v_next, weight)

        # Threshold
        if v_next >= threshold:
            spikes_out.append(t)
            v_next = 0

        total_toggles += _count_toggles(prev_v, v_next)
        prev_v = v_next
        v = v_next

    return total_toggles, spikes_out


@dataclass(frozen=True)
class ActivityRow:
    """Toggle counts of both models at one input activity."""

    activity_pct: float
    interval: int
    clock_toggles: int
    event_toggles: int
    clock_spikes: int
    event_spikes: int

    @property
    def reduction(self) -> float:
        """Return how many times fewer toggles the event-driven model makes."""
        return self.clock_toggles / max(self.event_toggles, 1)

    @property
    def fewer_toggles_pct(self) -> float:
        """Return the event-driven model's toggle count below the clock-driven one, in percent."""
        return 100.0 * (1.0 - self.event_toggles / max(self.clock_toggles, 1))


def compare(
    activities: Sequence[float] = DEFAULT_ACTIVITIES, n_cycles: int = DEFAULT_CYCLES
) -> list[ActivityRow]:
    """Run both models at each input activity and return their toggle counts.

    The event-driven model ticks its leak once per input interval, so both its
    input events and its leak ticks follow the activity.

    Parameters
    ----------
    activities : sequence of float
        Input activities in percent of cycles.
    n_cycles : int
        Number of clock cycles per run.

    Returns
    -------
    list of ActivityRow
        One row per activity, in the given order.
    """
    leak_k = _q88_encode(0.1)  # ~26 in Q8.8
    gain_k = _q88_encode(1.0)  # 256 in Q8.8
    threshold = _q88_encode(1.0)  # 256 in Q8.8
    rows = []
    for activity_pct in activities:
        interval = max(1, int(100 / activity_pct))
        spike_times = set(range(0, n_cycles, interval))
        t_clock, s_clock = simulate_clock_driven(n_cycles, spike_times, leak_k, gain_k, threshold)
        t_event, s_event = simulate_event_driven(
            n_cycles, spike_times, leak_k, threshold, leak_period=interval
        )
        rows.append(
            ActivityRow(activity_pct, interval, t_clock, t_event, len(s_clock), len(s_event))
        )
    return rows


def conclusion(rows: Sequence[ActivityRow], n_cycles: int = DEFAULT_CYCLES) -> list[str]:
    """Summarise measured rows without stating anything the rows do not show.

    Parameters
    ----------
    rows : sequence of ActivityRow
        Rows from :func:`compare`, from the highest activity to the lowest.
    n_cycles : int
        Number of clock cycles per run.

    Returns
    -------
    list of str
        Lines of the summary.
    """
    first, last = rows[0], rows[-1]
    reductions = [row.reduction for row in rows]
    return [
        f"Membrane-register toggles over {n_cycles:,} cycles, from "
        f"{first.activity_pct}% to {last.activity_pct}% activity:",
        f"clock-driven {first.clock_toggles:,} -> {last.clock_toggles:,}, "
        f"event-driven {first.event_toggles:,} -> {last.event_toggles:,}.",
        f"The event-driven model makes {min(reductions):.1f}x to "
        f"{max(reductions):.1f}x fewer toggles at these activities.",
        "Clock distribution, the leak counter and control logic are not",
        "counted, so this is not a power measurement.",
    ]


def main() -> None:
    """Print the toggle counts of both models and the derived summary."""
    print("=" * 60)
    print("  SC-NeuroCore: Clock-Driven vs Event-Driven Toggle Counts")
    print("=" * 60)

    rows = compare()
    for row in rows:
        print(
            f"\n  Activity: {row.activity_pct}% (1 input every {row.interval} cycles;"
            f" event leak tick every {row.interval} cycles)"
        )
        print(f"  {'-' * 50}")
        print(
            f"  Clock-driven:  {row.clock_toggles:>10,} toggles"
            f"  ({row.clock_toggles / DEFAULT_CYCLES:.2f}/cycle)"
        )
        print(
            f"  Event-driven:  {row.event_toggles:>10,} toggles"
            f"  ({row.event_toggles / DEFAULT_CYCLES:.2f}/cycle)"
        )
        print(f"  Toggle reduction: {row.reduction:.1f}x")
        print(f"  Fewer toggles:    {row.fewer_toggles_pct:.0f}%")
        print(f"  Output spikes:    clock={row.clock_spikes}, event={row.event_spikes}")

    print(f"\n{'=' * 60}")
    for line in conclusion(rows):
        print(f"  {line}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
