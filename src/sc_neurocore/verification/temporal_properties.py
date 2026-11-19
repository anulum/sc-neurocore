# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Temporal logic verification for SNNs

"""Verify temporal properties of spiking neural networks.

Specify safety/liveness properties over spike trains and check them on a
recorded spike raster. A check covers the trace it reads, not every input.

Every check reads a bounded ``(T, N)`` raster of non-negative spike counts. It returns ``UNKNOWN`` where the
question reaches past either end of the trace and the recorded part does not settle it, and refuses neuron ids,
stimulus times and window sizes that lie outside the trace. The short-trace rule of the window checks relies on the
counts being non-negative: every window holding the whole trace counts at least its spikes.

Properties:
  - fires_within: neuron responds within time window
  - mutual_exclusion: no two neurons fire simultaneously
  - rate_bound: firing rate stays below safety threshold
  - refractory_guarantee: minimum inter-spike interval
  - causal_order: neuron A fires before neuron B
  - bounded_activity: total spikes in window stay within bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class PropertyResult(Enum):
    """Outcome states returned by temporal property verification checks."""

    VERIFIED = "verified"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class Counterexample:
    """Input that violates a property."""

    timestep: int
    neuron_ids: list[int]
    description: str


@dataclass
class VerificationResult:
    """Result of a temporal property check."""

    property_name: str
    result: PropertyResult
    counterexample: Counterexample | None = None
    checked_steps: int = 0
    message: str = ""

    def summary(self) -> str:
        """Format the verification outcome and optional counterexample."""
        icon = {"verified": "PASS", "violated": "FAIL", "unknown": "?"}[self.result.value]
        line = f"[{icon}] {self.property_name}: {self.message}"
        if self.counterexample:
            line += f"\n  Counterexample at t={self.counterexample.timestep}: {self.counterexample.description}"
        return line


def _steps(spikes: np.ndarray[Any, Any], *neuron_ids: int) -> int:
    """Return the trace length after checking the raster's shape and the neuron ids.

    Raises
    ------
    ValueError
        If ``spikes`` is not two-dimensional or a neuron id lies outside ``[0, N)``.
    """
    if spikes.ndim != 2:
        raise ValueError(f"spikes must have shape (T, N), got shape {spikes.shape}")
    n_neurons = spikes.shape[1]
    for neuron in neuron_ids:
        if not 0 <= neuron < n_neurons:
            raise ValueError(f"neuron id {neuron} lies outside the raster's {n_neurons} neurons")
    return int(spikes.shape[0])


def _window(window_size: int) -> None:
    """Refuse a window that holds no step."""
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1 step, got {window_size}")


def fires_within(
    spikes: np.ndarray[Any, Any],
    neuron_id: int,
    stimulus_times: list[int],
    max_latency: int,
) -> VerificationResult:
    """Verify that neuron fires within max_latency steps of each stimulus.

    Parameters
    ----------
    spikes : ndarray of shape (T, N)
    neuron_id : int
    stimulus_times : list of int
        Timesteps when stimulus was applied.
    max_latency : int
        Maximum allowed response latency in timesteps; a spike at ``t + max_latency`` still counts.

    Returns
    -------
    VerificationResult
        ``VIOLATED`` with a counterexample when a whole window ``[t, t + max_latency]`` inside the trace shows no
        spike; otherwise ``UNKNOWN`` when a window runs past the trace end without a spike; otherwise ``VERIFIED``.

    Raises
    ------
    ValueError
        If the raster, the neuron id, a stimulus time (outside ``[0, T)``) or a negative latency is invalid.
    """
    T = _steps(spikes, neuron_id)
    if max_latency < 0:
        raise ValueError(f"max_latency must be non-negative, got {max_latency}")
    for t_stim in stimulus_times:
        if not 0 <= t_stim < T:
            raise ValueError(f"stimulus time {t_stim} lies outside the trace [0, {T})")
    undecided: list[int] = []
    for t_stim in stimulus_times:
        deadline = t_stim + max_latency
        fired = bool(np.any(spikes[t_stim : min(deadline + 1, T), neuron_id] > 0))
        if not fired and deadline >= T:
            undecided.append(t_stim)
        elif not fired:
            return VerificationResult(
                property_name="fires_within",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=t_stim,
                    neuron_ids=[neuron_id],
                    description=f"Neuron {neuron_id} did not fire within {max_latency} "
                    f"steps of stimulus at t={t_stim}",
                ),
                checked_steps=T,
                message=f"Neuron {neuron_id} failed to respond at t={t_stim}",
            )
    if undecided:
        return VerificationResult(
            property_name="fires_within",
            result=PropertyResult.UNKNOWN,
            checked_steps=T,
            message=f"No response of neuron {neuron_id} yet to the stimuli at {undecided}; their "
            f"deadlines fall after the trace ends at t={T - 1}",
        )

    return VerificationResult(
        property_name="fires_within",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"Neuron {neuron_id} responds within {max_latency} steps for all "
        f"{len(stimulus_times)} stimuli",
    )


def mutual_exclusion(
    spikes: np.ndarray[Any, Any],
    neuron_set: list[int],
) -> VerificationResult:
    """Verify that no two neurons in the set fire at the same timestep.

    Parameters
    ----------
    spikes : ndarray of shape (T, N)
    neuron_set : list of int
        Neuron IDs that should never co-fire.
    """
    T = _steps(spikes, *neuron_set)
    for t in range(T):
        active = [n for n in neuron_set if spikes[t, n] > 0]
        if len(active) > 1:
            return VerificationResult(
                property_name="mutual_exclusion",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=t,
                    neuron_ids=active,
                    description=f"Neurons {active} fire simultaneously at t={t}",
                ),
                checked_steps=T,
                message=f"Mutual exclusion violated at t={t}",
            )

    return VerificationResult(
        property_name="mutual_exclusion",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"No simultaneous firing among {len(neuron_set)} neurons over {T} steps",
    )


def rate_bound(
    spikes: np.ndarray[Any, Any],
    neuron_id: int,
    max_rate: float,
    window_size: int,
) -> VerificationResult:
    """Verify firing rate stays below max_rate in every sliding window.

    Parameters
    ----------
    spikes : ndarray of shape (T, N)
    neuron_id : int
    max_rate : float
        Maximum allowed firing rate (spikes per step).
    window_size : int
        Sliding window size in timesteps, at least 1. When the trace is shorter than one window, the bound is
        ``VIOLATED`` if the whole trace already exceeds it (every window holding the trace would too) and
        ``UNKNOWN`` otherwise.
    """
    T = _steps(spikes, neuron_id)
    _window(window_size)
    neuron_spikes = spikes[:, neuron_id]
    if window_size > T:
        rate = float(neuron_spikes.sum()) / window_size
        if rate <= max_rate:
            return VerificationResult(
                property_name="rate_bound",
                result=PropertyResult.UNKNOWN,
                checked_steps=T,
                message=f"Trace of {T} steps is shorter than one {window_size}-step window",
            )

    for t in range(max(T - window_size + 1, 1)):
        window = neuron_spikes[t : t + window_size]
        rate = float(window.sum()) / window_size
        if rate > max_rate:
            return VerificationResult(
                property_name="rate_bound",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=t,
                    neuron_ids=[neuron_id],
                    description=f"Rate {rate:.3f} > {max_rate:.3f} in window [{t}, {t + window_size})",
                ),
                checked_steps=T,
                message=f"Rate bound violated at t={t}: {rate:.3f} > {max_rate}",
            )

    return VerificationResult(
        property_name="rate_bound",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"Neuron {neuron_id} rate stays below {max_rate} in all {window_size}-step windows",
    )


def refractory_guarantee(
    spikes: np.ndarray[Any, Any],
    neuron_id: int,
    min_gap: int,
) -> VerificationResult:
    """Verify minimum inter-spike interval.

    Parameters
    ----------
    spikes : ndarray of shape (T, N)
    neuron_id : int
    min_gap : int
        Minimum required gap between consecutive spikes (timesteps).
    """
    T = _steps(spikes, neuron_id)
    spike_times = np.where(spikes[:, neuron_id] > 0)[0]

    for i in range(len(spike_times) - 1):
        gap = spike_times[i + 1] - spike_times[i]
        if gap < min_gap:
            return VerificationResult(
                property_name="refractory_guarantee",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=int(spike_times[i]),
                    neuron_ids=[neuron_id],
                    description=f"ISI = {gap} < {min_gap} between t={spike_times[i]} and t={spike_times[i + 1]}",
                ),
                checked_steps=T,
                message=f"Refractory violated: ISI={gap} at t={spike_times[i]}",
            )

    return VerificationResult(
        property_name="refractory_guarantee",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"All ISIs >= {min_gap} for neuron {neuron_id} ({len(spike_times)} spikes)",
    )


def causal_order(
    spikes: np.ndarray[Any, Any],
    neuron_a: int,
    neuron_b: int,
    max_delay: int,
) -> VerificationResult:
    """Verify that neuron A fires before neuron B within max_delay.

    For every spike of neuron B, there must be a spike of neuron A
    within the preceding max_delay timesteps.

    Parameters
    ----------
    spikes : ndarray of shape (T, N)
    neuron_a, neuron_b : int
    max_delay : int
        Largest allowed lag of B behind A, non-negative.

    Returns
    -------
    VerificationResult
        ``VIOLATED`` when a B spike's whole window ``[t - max_delay, t)`` lies inside the trace without an A spike;
        otherwise ``UNKNOWN`` when a B spike's window starts before ``t = 0`` and its recorded part holds no A spike
        (the cause may precede the recording); otherwise ``VERIFIED``.
    """
    T = _steps(spikes, neuron_a, neuron_b)
    if max_delay < 0:
        raise ValueError(f"max_delay must be non-negative, got {max_delay}")
    b_times = np.where(spikes[:, neuron_b] > 0)[0]
    a_times = set(np.where(spikes[:, neuron_a] > 0)[0].tolist())

    undecided: list[int] = []
    for t_b in b_times:
        found = False
        for dt in range(1, max_delay + 1):
            if (t_b - dt) in a_times:
                found = True
                break
        if not found and t_b < max_delay:
            undecided.append(int(t_b))
        elif not found:
            return VerificationResult(
                property_name="causal_order",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=int(t_b),
                    neuron_ids=[neuron_a, neuron_b],
                    description=f"Neuron {neuron_b} fired at t={t_b} without neuron {neuron_a} "
                    f"firing in [{t_b - max_delay}, {t_b})",
                ),
                checked_steps=T,
                message=f"Causal order violated at t={t_b}",
            )
    if undecided:
        return VerificationResult(
            property_name="causal_order",
            result=PropertyResult.UNKNOWN,
            checked_steps=T,
            message=f"No cause of neuron {neuron_b}'s spikes at {undecided} in the trace; their windows "
            "start before the trace does",
        )

    return VerificationResult(
        property_name="causal_order",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"Neuron {neuron_a} precedes neuron {neuron_b} within {max_delay} steps "
        f"for all {len(b_times)} B-spikes",
    )


def bounded_activity(
    spikes: np.ndarray[Any, Any],
    neuron_set: list[int],
    window_size: int,
    max_total_spikes: int,
) -> VerificationResult:
    """Verify total spike count in neuron set stays bounded per window.

    Parameters
    ----------
    spikes : ndarray of shape (T, N)
    neuron_set : list of int
    window_size : int
        Window size in timesteps, at least 1; a trace shorter than one window is judged as in
        :func:`rate_bound`.
    max_total_spikes : int
    """
    T = _steps(spikes, *neuron_set)
    _window(window_size)
    subset = spikes[:, neuron_set]
    if window_size > T and int(subset.sum()) <= max_total_spikes:
        return VerificationResult(
            property_name="bounded_activity",
            result=PropertyResult.UNKNOWN,
            checked_steps=T,
            message=f"Trace of {T} steps is shorter than one {window_size}-step window",
        )

    for t in range(max(T - window_size + 1, 1)):
        total = int(subset[t : t + window_size].sum())
        if total > max_total_spikes:
            return VerificationResult(
                property_name="bounded_activity",
                result=PropertyResult.VIOLATED,
                counterexample=Counterexample(
                    timestep=t,
                    neuron_ids=neuron_set,
                    description=f"Total spikes = {total} > {max_total_spikes} in "
                    f"window [{t}, {t + window_size})",
                ),
                checked_steps=T,
                message=f"Activity bound violated at t={t}: {total} > {max_total_spikes}",
            )

    return VerificationResult(
        property_name="bounded_activity",
        result=PropertyResult.VERIFIED,
        checked_steps=T,
        message=f"Activity stays below {max_total_spikes} in all {window_size}-step windows",
    )
