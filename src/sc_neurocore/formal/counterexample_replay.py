# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal counterexample replay utilities

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .network_properties import NetworkRateBound, NetworkRefractoryInvariant

SpikeSample = bool | int | Sequence[bool | int]


@dataclass(frozen=True, slots=True)
class RateBoundReplayResult:
    """Replay result for an aligned-window network rate-bound property."""

    violated: bool
    first_violation_cycle: int | None
    window_start_cycle: int | None
    observed_spikes: int
    cycles_checked: int


@dataclass(frozen=True, slots=True)
class RefractoryReplayResult:
    """Replay result for a monitored-output refractory invariant."""

    violated: bool
    first_violation_cycle: int | None
    trigger_cycle: int | None
    remaining_refractory_cycles: int
    cycles_checked: int


def replay_rate_bound_counterexample(
    spike_trace: Sequence[SpikeSample],
    rate_bound: NetworkRateBound,
) -> RateBoundReplayResult:
    """Replay a spike trace against the same aligned-window rate bound used by SVA."""
    current_count = 0
    window_start_cycle = 0

    for cycle, sample in enumerate(spike_trace):
        if cycle > 0 and cycle % rate_bound.window_cycles == 0:
            current_count = 0
            window_start_cycle = cycle

        spike = _select_binary_spike(sample, rate_bound.output_index, cycle=cycle)
        current_count += int(spike)
        if current_count > rate_bound.max_spikes:
            return RateBoundReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                window_start_cycle=window_start_cycle,
                observed_spikes=current_count,
                cycles_checked=cycle + 1,
            )

    return RateBoundReplayResult(
        violated=False,
        first_violation_cycle=None,
        window_start_cycle=None,
        observed_spikes=current_count,
        cycles_checked=len(spike_trace),
    )


def replay_refractory_counterexample(
    spike_trace: Sequence[SpikeSample],
    refractory: NetworkRefractoryInvariant,
) -> RefractoryReplayResult:
    """Replay a spike trace against a monitored-output refractory invariant."""
    remaining = 0
    trigger_cycle: int | None = None

    for cycle, sample in enumerate(spike_trace):
        spike = _select_binary_spike(sample, refractory.output_index, cycle=cycle)
        if spike and remaining > 0:
            return RefractoryReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                trigger_cycle=trigger_cycle,
                remaining_refractory_cycles=remaining,
                cycles_checked=cycle + 1,
            )
        if spike:
            remaining = refractory.refractory_cycles
            trigger_cycle = cycle
        elif remaining > 0:
            remaining -= 1
            if remaining == 0:
                trigger_cycle = None

    return RefractoryReplayResult(
        violated=False,
        first_violation_cycle=None,
        trigger_cycle=None,
        remaining_refractory_cycles=remaining,
        cycles_checked=len(spike_trace),
    )


def _select_binary_spike(sample: SpikeSample, output_index: int, *, cycle: int) -> bool:
    if isinstance(sample, (bool, int)):
        if output_index != 0:
            raise ValueError("scalar spike trace samples only support output_index 0")
        return _as_binary_bool(sample, cycle=cycle)

    if isinstance(sample, (str, bytes)) or not isinstance(sample, Sequence):
        raise ValueError(f"cycle {cycle} must contain a binary spike sample")
    if output_index >= len(sample):
        raise ValueError(f"cycle {cycle} does not contain output_index {output_index}")
    return _as_binary_bool(sample[output_index], cycle=cycle)


def _as_binary_bool(value: bool | int, *, cycle: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"cycle {cycle} contains a non-binary spike value")
