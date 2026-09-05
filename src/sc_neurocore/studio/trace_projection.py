# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio raw trace custody and display projection

"""Raw per-step trace custody and the separate viewport projection.

A Studio run returns its complete raw result (every recorded variable at every
step, the drive, the spike steps and exact initial/final snapshots) under the
post-step observation clock, and *separately* a display projection bounded by
:data:`MAX_PLOT_POINTS`. The projection keeps, for every scalar trace and the
drive, the minimum and the maximum of each bucket plus the first and the last
sample, on one shared sample-index set, so peaks, resets and the final sample
survive reduction and every display point maps back to a raw step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sc_neurocore.studio.state_layout import StateLayout, public_snapshot

MAX_PLOT_POINTS = 5_000
RAW_ELEMENT_BUDGET = 2_000_000
RAW_SCHEMA_VERSION = "studio.raw-trace.v1"
DISPLAY_SCHEMA_VERSION = "studio.display-projection.v1"
OBSERVATION_CLOCK = "post-step"
SAMPLE_TIME_RULE = "(index + 1) * dt"
DRIVE_INTERVAL_RULE = "[index * dt, (index + 1) * dt)"

ProjectionMethod = Literal["identity", "bucket-extrema"]


@dataclass(frozen=True, slots=True)
class DisplayProjection:
    """The sample-index set of the viewport projection.

    Parameters
    ----------
    sample_index : numpy.ndarray
        Sorted raw step indices the projection shows.
    method : {"identity", "bucket-extrema"}
        ``identity`` when the run fits the point budget, otherwise per-bucket
        extrema of every series on a shared index set.
    bucket_count : int
        Number of buckets used (``0`` for identity).
    max_points : int
        The upper bound the projection was built against.
    """

    sample_index: np.ndarray[Any, Any]
    method: ProjectionMethod
    bucket_count: int
    max_points: int

    @property
    def point_count(self) -> int:
        """Number of display points."""
        return int(self.sample_index.size)


def display_sample_indices(
    n_steps: int,
    series: Sequence[np.ndarray[Any, Any]],
    *,
    max_points: int = MAX_PLOT_POINTS,
) -> DisplayProjection:
    """Choose the raw step indices a bounded viewport shows.

    Parameters
    ----------
    n_steps : int
        Number of raw steps.
    series : sequence of numpy.ndarray
        Every scalar trace whose extrema must survive (states and drive), each
        of length ``n_steps`` and finite.
    max_points : int
        Hard upper bound on the number of display points.

    Returns
    -------
    DisplayProjection
        Identity when ``n_steps <= max_points``; otherwise the sorted union of
        the first sample, the last sample and each bucket's argmin/argmax of
        every series. With ``k`` series and ``b`` buckets the union holds at
        most ``2 + 2*k*b <= max_points`` points, so the bound is exact.

    Raises
    ------
    ValueError
        When ``n_steps`` is not positive, ``max_points < 2`` or a series has
        the wrong length, contains non-finite values, or the point budget
        cannot guarantee the endpoints and every series' extrema.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    for values in series:
        if values.shape != (n_steps,):
            raise ValueError(f"series shape {values.shape} does not match n_steps {n_steps}")
        if not np.isfinite(values).all():
            raise ValueError("display series must contain only finite values")
    if n_steps <= max_points:
        return DisplayProjection(
            sample_index=np.arange(n_steps, dtype=np.int64),
            method="identity",
            bucket_count=0,
            max_points=max_points,
        )
    k = max(1, len(series))
    if series and max_points < 2 + 2 * k:
        raise ValueError("max_points cannot preserve endpoints and every series' extrema")
    buckets = max(1, (max_points - 2) // (2 * k))
    edges = np.linspace(0, n_steps, buckets + 1).astype(np.int64)
    chosen: set[int] = {0, n_steps - 1}
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue
        for values in series:
            segment = values[start:stop]
            chosen.add(int(start + np.argmin(segment)))
            chosen.add(int(start + np.argmax(segment)))
    fill = max_points - len(chosen)
    if fill > 0:
        # Extrema often coincide across series; spend the unused budget on
        # evenly spaced samples so the projection also follows slow drifts.
        chosen.update(int(i) for i in np.linspace(0, n_steps - 1, fill).astype(np.int64))
    sample_index = np.array(sorted(chosen), dtype=np.int64)
    if sample_index.size > max_points:  # pragma: no cover - guarded by the bound above
        raise AssertionError("display projection exceeded its bound")
    return DisplayProjection(
        sample_index=sample_index,
        method="bucket-extrema",
        bucket_count=buckets,
        max_points=max_points,
    )


def sample_times(n_steps: int, dt: float) -> np.ndarray[Any, Any]:
    """Return the post-step sample time of every raw step: ``(index + 1) * dt``."""
    return (np.arange(n_steps, dtype=np.float64) + 1.0) * dt


def observation_block(dt: float) -> dict[str, object]:
    """Return the observation-clock contract of a run."""
    return {
        "clock": OBSERVATION_CLOCK,
        "dt": dt,
        "initial_time_ms": 0.0,
        "sample_time_ms": SAMPLE_TIME_RULE,
        "drive_interval_ms": DRIVE_INTERVAL_RULE,
    }


def raw_block(
    *,
    dt: float,
    n_steps: int,
    scalar_traces: Mapping[str, np.ndarray[Any, Any]],
    vector_traces: Mapping[str, np.ndarray[Any, Any]],
    vector_omitted: Sequence[str],
    drive: np.ndarray[Any, Any],
    spikes: Sequence[int],
    element_budget: int = RAW_ELEMENT_BUDGET,
) -> dict[str, object]:
    """Return the full-resolution raw block of a run.

    Parameters
    ----------
    dt, n_steps : float, int
        Step and number of executed steps.
    scalar_traces : mapping
        Per-step float64 arrays of length ``n_steps`` for every scalar state.
    vector_traces : mapping
        Per-step arrays of shape ``(n_steps, *shape)`` for vector states that
        fit the element budget.
    vector_omitted : sequence of str
        Vector states recorded in snapshots only.
    drive : numpy.ndarray
        The injected drive sample of every step.
    spikes : sequence of int
        Raw step indices at which the model reported a spike.
    element_budget : int
        Raw element budget the block was built against.

    Returns
    -------
    dict
        ``included`` is ``False`` only when the scalar traces alone exceed the
        budget; nothing is shortened silently, the reason is stated instead.
    """
    scalar_elements = n_steps * (len(scalar_traces) + 1)
    vector_elements = sum(int(values.size) for values in vector_traces.values())
    elements = scalar_elements + vector_elements
    included = scalar_elements <= element_budget
    block: dict[str, object] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "included": included,
        "element_count": elements,
        "element_budget": element_budget,
        "dt": dt,
        "n_steps": n_steps,
        "sample_time_ms": SAMPLE_TIME_RULE,
        "drive_interval_ms": DRIVE_INTERVAL_RULE,
        "spike_indices": [int(index) for index in spikes],
        "spike_times_ms": [float((int(index) + 1) * dt) for index in spikes],
        "vector_snapshots_only": list(vector_omitted),
    }
    if included:
        block["states"] = {name: values.tolist() for name, values in scalar_traces.items()}
        block["vector_states"] = {name: values.tolist() for name, values in vector_traces.items()}
        block["drive"] = drive.tolist()
    else:
        block["reason"] = (
            f"raw traces need {scalar_elements} elements, above the budget of "
            f"{element_budget}; reduce the duration or the number of recorded states"
        )
    return block


def custody_payload(
    *,
    dt: float,
    n_steps: int,
    layout: StateLayout,
    initial_state: Mapping[str, float | np.ndarray[Any, Any]],
    final_state: Mapping[str, float | np.ndarray[Any, Any]],
    scalar_traces: Mapping[str, np.ndarray[Any, Any]],
    vector_traces: Mapping[str, np.ndarray[Any, Any]],
    vector_omitted: Sequence[str],
    drive: np.ndarray[Any, Any],
    spikes: Sequence[int],
    stats: Mapping[str, object],
    max_points: int = MAX_PLOT_POINTS,
    element_budget: int = RAW_ELEMENT_BUDGET,
) -> dict[str, Any]:
    """Assemble the public result of a run: raw custody plus display projection.

    The top-level ``time``, ``states`` and ``current_trace`` fields are the
    display projection (kept for existing consumers and labelled as such in
    ``display``); ``raw``, ``initial_state`` and ``final_state`` carry the
    complete result.
    """
    series = [*scalar_traces.values(), drive]
    projection = display_sample_indices(n_steps, series, max_points=max_points)
    index = projection.sample_index
    times = sample_times(n_steps, dt)
    return {
        "time": times[index].tolist(),
        "states": {name: values[index].tolist() for name, values in scalar_traces.items()},
        "current_trace": drive[index].tolist(),
        "spikes": [int(step) for step in spikes],
        "spike_count": len(spikes),
        "stats": dict(stats),
        "dt": dt,
        "n_steps": n_steps,
        "observation": observation_block(dt),
        "state_layout": layout.to_public_dict(),
        "initial_state": public_snapshot(initial_state),
        "final_state": public_snapshot(final_state),
        "raw": raw_block(
            dt=dt,
            n_steps=n_steps,
            scalar_traces=scalar_traces,
            vector_traces=vector_traces,
            vector_omitted=vector_omitted,
            drive=drive,
            spikes=spikes,
            element_budget=element_budget,
        ),
        "display": {
            "schema_version": DISPLAY_SCHEMA_VERSION,
            "method": projection.method,
            "max_points": projection.max_points,
            "bucket_count": projection.bucket_count,
            "point_count": projection.point_count,
            "sample_index": index.tolist(),
            "first_sample_included": True,
            "final_sample_included": True,
            "spikes_are_raw_steps": True,
        },
    }


def full_state_traces(result: Mapping[str, Any]) -> dict[str, list[float]]:
    """Return the full-resolution scalar state traces of a run result.

    Consumers that compute on a trace (attractor detection, state ranges,
    precision errors) must not use the display projection. When the result
    carries an included ``raw`` block its traces are returned; a legacy or
    raw-less result falls back to its ``states`` field.
    """
    raw = result.get("raw")
    if isinstance(raw, Mapping) and raw.get("included"):
        states = raw.get("states")
        if isinstance(states, Mapping):
            return {str(name): list(values) for name, values in states.items()}
    states = result.get("states")
    if isinstance(states, Mapping):
        return {str(name): list(values) for name, values in states.items()}
    return {}


def full_state_trace(result: Mapping[str, Any], name: str) -> list[float]:
    """Return one full-resolution scalar state trace (empty when absent)."""
    return full_state_traces(result).get(name, [])


__all__ = [
    "DISPLAY_SCHEMA_VERSION",
    "DRIVE_INTERVAL_RULE",
    "MAX_PLOT_POINTS",
    "OBSERVATION_CLOCK",
    "RAW_ELEMENT_BUDGET",
    "RAW_SCHEMA_VERSION",
    "SAMPLE_TIME_RULE",
    "DisplayProjection",
    "ProjectionMethod",
    "custody_payload",
    "display_sample_indices",
    "full_state_trace",
    "full_state_traces",
    "observation_block",
    "raw_block",
    "sample_times",
]
