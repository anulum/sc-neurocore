# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained trajectories for repeated runtime measurements

"""Retain every timed trajectory before a runtime can reuse its output buffers."""

from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict

from typing_extensions import NotRequired
import time

import numpy as np
import numpy.typing as npt


def measure_repetitions(
    simulate: Callable[..., Mapping[str, object]],
    currents: npt.NDArray[np.float64],
    *,
    backend: str,
    repeats: int,
) -> tuple[list[int], list[Mapping[str, object]]]:
    """Measure complete runs and retain independent arrays for every result.

    Parameters
    ----------
    simulate : callable
        Maintained model dispatcher accepting currents and an explicit backend.
    currents : numpy.ndarray
        Complete current vector for each timed run.
    backend : str
        Actual runtime selected by the dispatcher.
    repeats : int
        Positive number of timed runs after the existing bounded warmup.

    Returns
    -------
    tuple
        Elapsed nanoseconds and result snapshots in matching repetition order.
        Copying output arrays happens after the elapsed time is recorded.

    Raises
    ------
    ValueError
        If the requested repetition count is not positive.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    simulate(currents[: min(1000, len(currents))], backend=backend)
    samples: list[int] = []
    results: list[Mapping[str, object]] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = simulate(currents, backend=backend)
        samples.append(time.perf_counter_ns() - started)
        results.append(
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in result.items()
            }
        )
    return samples, results


class RepetitionSummary(TypedDict):
    """Timing and trajectory parity for one runtime or one repetition."""

    available: bool
    samples_ns: list[int]
    median_ns_per_step: float
    min_ns_per_step: float
    max_ns_per_step: float
    trace_sha256: str | None
    parity_atol: float
    parity_max_abs_diff: float
    trace_matches_python: bool
    events: int
    spikes: int
    event_vector_matches_python: bool
    final_state: list[float]
    repetitions: NotRequired[list["RepetitionSummary"]]


def summarize_repetitions(
    samples: list[int],
    results: list[Mapping[str, object]],
    summarize: Callable[[list[int], Mapping[str, object]], RepetitionSummary],
) -> RepetitionSummary:
    """Keep every trajectory verdict beside aggregate timing and parity.

    Parameters
    ----------
    samples : list of int
        Positive elapsed nanoseconds in measurement order.
    results : list of mapping
        Independent trajectories in the same repetition order.
    summarize : callable
        Model-specific comparison against its fixed Python reference.

    Returns
    -------
    RepetitionSummary
        Aggregate timing, last trajectory identity, all individual comparisons,
        and the worst parity verdict across the complete repetition cohort.

    Raises
    ------
    ValueError
        If measurements are empty or sample and trajectory counts differ.
    """
    if not samples or len(samples) != len(results):
        raise ValueError("each timed repetition requires one retained trajectory")
    row = summarize(samples, results[-1])
    checks = [summarize([sample], result) for sample, result in zip(samples, results, strict=True)]
    row["repetitions"] = checks
    row["parity_max_abs_diff"] = max(check["parity_max_abs_diff"] for check in checks)
    row["event_vector_matches_python"] = all(
        check["event_vector_matches_python"] for check in checks
    )
    row["trace_matches_python"] = all(check["trace_matches_python"] for check in checks)
    return row


def compare_trajectories(
    result: Mapping[str, object],
    reference: Mapping[str, object],
    state_keys: Sequence[str],
    tolerance: float,
) -> tuple[float, bool, bool]:
    """Compare complete finite state vectors and exact discrete events.

    Parameters
    ----------
    result, reference : mapping
        Measured and Python-reference trajectories with matching state keys.
    state_keys : sequence of str
        Continuous state vectors required by the model recurrence.
    tolerance : float
        Maintained backend's absolute state tolerance.

    Returns
    -------
    tuple
        Maximum absolute state error, exact event agreement and full parity.
        Invalid state shape or non-finite values produce infinite error and
        failed parity, even when another state vector matches exactly.
    """
    maximum = 0.0
    for key in state_keys:
        actual = np.asarray(result[key], dtype=np.float64)
        expected = np.asarray(reference[key], dtype=np.float64)
        if (
            actual.ndim != 1
            or actual.shape != expected.shape
            or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
        ):
            maximum = float("inf")
            break
        maximum = max(maximum, float(np.abs(actual - expected).max(initial=0.0)))
    events_match = np.array_equal(np.asarray(result["events"]), np.asarray(reference["events"]))
    return maximum, events_match, maximum <= tolerance and events_match
