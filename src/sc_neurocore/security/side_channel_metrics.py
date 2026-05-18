# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Analytic side-channel proxy metrics for stochastic-computing bitstreams.

These helpers quantify switching activity in simulated or replayed bitstreams.
They are not physical leakage measurements and do not claim DPA resistance or
board-level power/thermal security.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class SideChannelMetricError(ValueError):
    """Raised when side-channel metric inputs are malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class SwitchingActivitySummary:
    """Transition-count summary for a rectangular binary bitstream matrix."""

    cycles: int
    stream_count: int
    per_stream_transition_counts: tuple[int, ...]
    per_stream_transition_rates: tuple[float, ...]
    mean_transition_rate: float
    max_transition_rate: float
    total_transitions: int
    activity_histogram: dict[int, int]


@dataclass(frozen=True, slots=True)
class ClassActivityProxy:
    """Class-conditioned analytic proxy for activity-dependent leakage.

    ``label_activity_correlation`` is Pearson correlation between numeric labels
    and per-sample mean switching rate. It is ``None`` when either side has zero
    variance, avoiding fabricated correlation claims.
    """

    class_means: dict[int | float, float]
    max_class_mean_gap: float
    label_activity_correlation: float | None
    sample_count: int


def compute_switching_activity(
    bitstreams: Sequence[Sequence[int]],
) -> SwitchingActivitySummary:
    """Compute per-stream switching activity for rows of binary bitstreams."""

    matrix = _normalise_bitstream_matrix(bitstreams)
    cycles = len(matrix[0])
    transition_counts = tuple(
        sum(1 for index in range(1, len(row)) if row[index - 1] != row[index])
        for row in matrix
    )
    denominator = cycles - 1
    transition_rates = tuple(count / denominator for count in transition_counts)
    histogram: dict[int, int] = {}
    for count in transition_counts:
        histogram[count] = histogram.get(count, 0) + 1

    return SwitchingActivitySummary(
        cycles=cycles,
        stream_count=len(matrix),
        per_stream_transition_counts=transition_counts,
        per_stream_transition_rates=transition_rates,
        mean_transition_rate=sum(transition_rates) / len(transition_rates),
        max_transition_rate=max(transition_rates),
        total_transitions=sum(transition_counts),
        activity_histogram=dict(sorted(histogram.items())),
    )


def compute_class_activity_proxy(
    bitstreams_by_sample: Sequence[Sequence[Sequence[int]]],
    labels: Sequence[int | float],
) -> ClassActivityProxy:
    """Summarise class-conditioned switching activity for simulated samples."""

    samples = _normalise_sample_collection(bitstreams_by_sample)
    label_values = _normalise_labels(labels)
    if len(samples) != len(label_values):
        raise SideChannelMetricError("bitstreams_by_sample and labels must have equal length")

    sample_rates = tuple(
        compute_switching_activity(sample).mean_transition_rate for sample in samples
    )
    rates_by_label: dict[int | float, list[float]] = {}
    for label, rate in zip(label_values, sample_rates, strict=True):
        rates_by_label.setdefault(label, []).append(rate)

    class_means = {
        label: sum(rates) / len(rates) for label, rates in sorted(rates_by_label.items())
    }
    mean_values = tuple(class_means.values())
    max_gap = max(mean_values) - min(mean_values) if len(mean_values) > 1 else 0.0

    return ClassActivityProxy(
        class_means=class_means,
        max_class_mean_gap=max_gap,
        label_activity_correlation=_pearson_correlation(label_values, sample_rates),
        sample_count=len(samples),
    )


def _normalise_sample_collection(
    bitstreams_by_sample: Sequence[Sequence[Sequence[int]]],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not isinstance(bitstreams_by_sample, Sequence) or not bitstreams_by_sample:
        raise SideChannelMetricError("bitstreams_by_sample must not be empty")
    return tuple(_normalise_bitstream_matrix(sample) for sample in bitstreams_by_sample)


def _normalise_bitstream_matrix(
    bitstreams: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(bitstreams, Sequence) or not bitstreams:
        raise SideChannelMetricError("bitstreams must be a non-empty rectangular matrix")

    rows: list[tuple[int, ...]] = []
    expected_cycles: int | None = None
    for row in bitstreams:
        if not isinstance(row, Sequence) or isinstance(row, (bytes, str)):
            raise SideChannelMetricError("each bitstream row must be a sequence")
        values = tuple(_normalise_bit(value) for value in row)
        if len(values) < 2:
            raise SideChannelMetricError("each bitstream row must contain at least two cycles")
        if expected_cycles is None:
            expected_cycles = len(values)
        elif len(values) != expected_cycles:
            raise SideChannelMetricError("bitstream matrix must be rectangular")
        rows.append(values)

    return tuple(rows)


def _normalise_bit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise SideChannelMetricError("bitstreams must contain only integer 0/1 values")
    return value


def _normalise_labels(labels: Sequence[int | float]) -> tuple[int | float, ...]:
    if not isinstance(labels, Sequence) or not labels:
        raise SideChannelMetricError("labels must not be empty")

    normalised: list[int | float] = []
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int | float):
            raise SideChannelMetricError("labels must be finite numeric values")
        if not math.isfinite(float(label)):
            raise SideChannelMetricError("labels must be finite numeric values")
        normalised.append(label)
    return tuple(normalised)


def _pearson_correlation(
    labels: Sequence[int | float],
    rates: Sequence[float],
) -> float | None:
    label_mean = sum(float(label) for label in labels) / len(labels)
    rate_mean = sum(rates) / len(rates)
    label_delta = tuple(float(label) - label_mean for label in labels)
    rate_delta = tuple(rate - rate_mean for rate in rates)
    label_ss = sum(delta * delta for delta in label_delta)
    rate_ss = sum(delta * delta for delta in rate_delta)
    if label_ss == 0.0 or rate_ss == 0.0:
        return None
    return sum(
        left * right for left, right in zip(label_delta, rate_delta, strict=True)
    ) / math.sqrt(label_ss * rate_ss)
