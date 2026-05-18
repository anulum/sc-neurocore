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

from .network_properties import (
    NetworkAntagonisticOutputExclusion,
    NetworkOutputTemporalSeparation,
    NetworkPopulationCoactivationCap,
    NetworkPopulationInactivityBound,
    NetworkPopulationSilenceAfterCoactivation,
    NetworkRateBound,
    NetworkRefractoryInvariant,
)

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


@dataclass(frozen=True, slots=True)
class AntagonisticReplayResult:
    """Replay result for a mutually-exclusive output-pair invariant."""

    violated: bool
    first_violation_cycle: int | None
    output_a: int
    output_b: int
    cycles_checked: int


@dataclass(frozen=True, slots=True)
class TemporalSeparationReplayResult:
    """Replay result for a bidirectional temporal-separation invariant."""

    violated: bool
    first_violation_cycle: int | None
    trigger_output: int | None
    violating_output: int | None
    remaining_separation_cycles: int
    cycles_checked: int


@dataclass(frozen=True, slots=True)
class PopulationCoactivationReplayResult:
    """Replay result for a population-level output coactivation cap."""

    violated: bool
    first_violation_cycle: int | None
    observed_active_outputs: int
    max_active_outputs: int
    cycles_checked: int


@dataclass(frozen=True, slots=True)
class PopulationSilenceReplayResult:
    """Replay result for post-coactivation global output silence."""

    violated: bool
    first_violation_cycle: int | None
    trigger_cycle: int | None
    observed_active_outputs: int
    remaining_silence_cycles: int
    trigger_active_outputs: int
    silence_cycles: int
    cycles_checked: int


@dataclass(frozen=True, slots=True)
class PopulationInactivityReplayResult:
    """Replay result for bounded consecutive population inactivity."""

    violated: bool
    first_violation_cycle: int | None
    observed_silent_cycles: int
    max_silent_cycles: int
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


def replay_antagonistic_counterexample(
    spike_trace: Sequence[SpikeSample],
    exclusion: NetworkAntagonisticOutputExclusion,
) -> AntagonisticReplayResult:
    """Replay a spike trace against a mutually-exclusive output-pair invariant."""
    for cycle, sample in enumerate(spike_trace):
        spike_a = _select_binary_spike(sample, exclusion.output_a, cycle=cycle)
        spike_b = _select_binary_spike(sample, exclusion.output_b, cycle=cycle)
        if spike_a and spike_b:
            return AntagonisticReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                output_a=exclusion.output_a,
                output_b=exclusion.output_b,
                cycles_checked=cycle + 1,
            )

    return AntagonisticReplayResult(
        violated=False,
        first_violation_cycle=None,
        output_a=exclusion.output_a,
        output_b=exclusion.output_b,
        cycles_checked=len(spike_trace),
    )


def replay_temporal_separation_counterexample(
    spike_trace: Sequence[SpikeSample],
    separation: NetworkOutputTemporalSeparation,
) -> TemporalSeparationReplayResult:
    """Replay a spike trace against a bidirectional output temporal separation."""
    remaining_after_a = 0
    remaining_after_b = 0

    for cycle, sample in enumerate(spike_trace):
        spike_a = _select_binary_spike(sample, separation.output_a, cycle=cycle)
        spike_b = _select_binary_spike(sample, separation.output_b, cycle=cycle)
        if spike_a and spike_b:
            return TemporalSeparationReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                trigger_output=None,
                violating_output=None,
                remaining_separation_cycles=0,
                cycles_checked=cycle + 1,
            )
        if spike_a and remaining_after_b > 0:
            return TemporalSeparationReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                trigger_output=separation.output_b,
                violating_output=separation.output_a,
                remaining_separation_cycles=remaining_after_b,
                cycles_checked=cycle + 1,
            )
        if spike_b and remaining_after_a > 0:
            return TemporalSeparationReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                trigger_output=separation.output_a,
                violating_output=separation.output_b,
                remaining_separation_cycles=remaining_after_a,
                cycles_checked=cycle + 1,
            )

        if spike_a:
            remaining_after_a = separation.separation_cycles
        elif remaining_after_a > 0:
            remaining_after_a -= 1
        if spike_b:
            remaining_after_b = separation.separation_cycles
        elif remaining_after_b > 0:
            remaining_after_b -= 1

    return TemporalSeparationReplayResult(
        violated=False,
        first_violation_cycle=None,
        trigger_output=None,
        violating_output=None,
        remaining_separation_cycles=max(remaining_after_a, remaining_after_b),
        cycles_checked=len(spike_trace),
    )


def replay_population_coactivation_counterexample(
    spike_trace: Sequence[SpikeSample],
    population: NetworkPopulationCoactivationCap,
) -> PopulationCoactivationReplayResult:
    """Replay a spike trace against a population coactivation cap."""
    max_observed = 0
    for cycle, sample in enumerate(spike_trace):
        active_outputs = _count_binary_spikes(sample, cycle=cycle)
        max_observed = max(max_observed, active_outputs)
        if active_outputs > population.max_active_outputs:
            return PopulationCoactivationReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                observed_active_outputs=active_outputs,
                max_active_outputs=population.max_active_outputs,
                cycles_checked=cycle + 1,
            )

    return PopulationCoactivationReplayResult(
        violated=False,
        first_violation_cycle=None,
        observed_active_outputs=max_observed,
        max_active_outputs=population.max_active_outputs,
        cycles_checked=len(spike_trace),
    )


def replay_population_silence_counterexample(
    spike_trace: Sequence[SpikeSample],
    silence: NetworkPopulationSilenceAfterCoactivation,
) -> PopulationSilenceReplayResult:
    """Replay a spike trace against a post-coactivation global silence contract."""
    remaining = 0
    trigger_cycle: int | None = None

    for cycle, sample in enumerate(spike_trace):
        active_outputs = _count_binary_spikes(sample, cycle=cycle)
        if remaining > 0 and active_outputs > 0:
            return PopulationSilenceReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                trigger_cycle=trigger_cycle,
                observed_active_outputs=active_outputs,
                remaining_silence_cycles=remaining,
                trigger_active_outputs=silence.trigger_active_outputs,
                silence_cycles=silence.silence_cycles,
                cycles_checked=cycle + 1,
            )
        if active_outputs >= silence.trigger_active_outputs:
            remaining = silence.silence_cycles
            trigger_cycle = cycle
        elif remaining > 0:
            remaining -= 1
            if remaining == 0:
                trigger_cycle = None

    return PopulationSilenceReplayResult(
        violated=False,
        first_violation_cycle=None,
        trigger_cycle=trigger_cycle if remaining > 0 else None,
        observed_active_outputs=0,
        remaining_silence_cycles=remaining,
        trigger_active_outputs=silence.trigger_active_outputs,
        silence_cycles=silence.silence_cycles,
        cycles_checked=len(spike_trace),
    )


def replay_population_inactivity_counterexample(
    spike_trace: Sequence[SpikeSample],
    inactivity: NetworkPopulationInactivityBound,
) -> PopulationInactivityReplayResult:
    """Replay a spike trace against a bounded consecutive-inactivity contract."""
    silent_run = 0
    max_observed_silent_run = 0

    for cycle, sample in enumerate(spike_trace):
        active_outputs = _count_binary_spikes(sample, cycle=cycle)
        if active_outputs == 0:
            silent_run += 1
            max_observed_silent_run = max(max_observed_silent_run, silent_run)
        else:
            silent_run = 0
        if silent_run > inactivity.max_silent_cycles:
            return PopulationInactivityReplayResult(
                violated=True,
                first_violation_cycle=cycle,
                observed_silent_cycles=silent_run,
                max_silent_cycles=inactivity.max_silent_cycles,
                cycles_checked=cycle + 1,
            )

    return PopulationInactivityReplayResult(
        violated=False,
        first_violation_cycle=None,
        observed_silent_cycles=max_observed_silent_run,
        max_silent_cycles=inactivity.max_silent_cycles,
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


def _count_binary_spikes(sample: SpikeSample, *, cycle: int) -> int:
    if isinstance(sample, (bool, int)):
        return int(_as_binary_bool(sample, cycle=cycle))

    if isinstance(sample, (str, bytes)) or not isinstance(sample, Sequence):
        raise ValueError(f"cycle {cycle} must contain a binary spike sample")
    return sum(int(_as_binary_bool(value, cycle=cycle)) for value in sample)


def _as_binary_bool(value: bool | int, *, cycle: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"cycle {cycle} contains a non-binary spike value")
