# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Activity-shaped stochastic encoding for analytic side-channel studies.

The encoders in this module preserve stochastic-computing probability counts
while distributing transitions across the bitstream. They emit analytic
simulation evidence only; they are not physical power, thermal, or DPA
resistance measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .side_channel_metrics import (
    ClassActivityProxy,
    SwitchingActivitySummary,
    compute_class_activity_proxy,
    compute_switching_activity,
)


class ThermalSCEncodingError(ValueError):
    """Raised when thermal side-channel encoder inputs violate the contract."""


@dataclass(frozen=True, slots=True)
class ThermalSCEncodingConfig:
    """Configuration for deterministic activity-balanced SC encoding."""

    bitstream_length: int = 256
    seed: int = 0
    rotation_stride: int = 1
    dummy_streams_per_record: int = 0
    max_dummy_overhead_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class ActivityBalancedEncoding:
    """A single activity-shaped stochastic-computing bitstream."""

    probability: float
    realised_probability: float
    bitstream: tuple[int, ...]
    dummy_bitstreams: tuple[tuple[int, ...], ...]
    seed_domain: str
    dummy_streams_inserted: int
    activity_summary: SwitchingActivitySummary
    evidence_boundary: str = "analytic_simulation_only"


@dataclass(frozen=True, slots=True)
class ActivityBalancedEncodingSummary:
    """Batch-level analytic evidence for activity-shaped encodings."""

    record_count: int
    class_activity_proxy: ClassActivityProxy
    total_dummy_streams_inserted: int
    max_dummy_streams_inserted: int
    dummy_stream_overhead_ratio: float
    evidence_boundary: str = "analytic_simulation_only"


@dataclass(frozen=True, slots=True)
class ActivityBalancedEncodingBatch:
    """Batch result for activity-shaped stochastic encodings."""

    records: tuple[ActivityBalancedEncoding, ...]
    summary: ActivityBalancedEncodingSummary


def encode_activity_balanced_probability(
    probability: float,
    config: ThermalSCEncodingConfig | None = None,
    *,
    stream_index: int = 0,
) -> ActivityBalancedEncoding:
    """Encode one probability with distributed ones and deterministic rotation."""

    cfg = _validate_config(config or ThermalSCEncodingConfig())
    probability_value = _validate_probability(probability)
    if not isinstance(stream_index, int) or stream_index < 0:
        raise ThermalSCEncodingError("stream_index must be a non-negative integer")

    ones = round(probability_value * cfg.bitstream_length)
    base = _distribute_ones(ones, cfg.bitstream_length)
    rotation = _activity_preserving_rotation_offset(base, cfg, stream_index)
    bitstream = _rotate(base, rotation)
    dummy_bitstreams = _build_dummy_bitstreams(cfg, stream_index)
    activity = compute_switching_activity((bitstream, *dummy_bitstreams))

    return ActivityBalancedEncoding(
        probability=probability_value,
        realised_probability=ones / cfg.bitstream_length,
        bitstream=bitstream,
        dummy_bitstreams=dummy_bitstreams,
        seed_domain=f"seed={cfg.seed}:stream={stream_index}:rotation={rotation}",
        dummy_streams_inserted=len(dummy_bitstreams),
        activity_summary=activity,
    )


def encode_activity_balanced_probabilities(
    probabilities: Sequence[float],
    config: ThermalSCEncodingConfig | None = None,
    *,
    labels: Sequence[int | float] | None = None,
) -> ActivityBalancedEncodingBatch:
    """Encode a probability batch and attach analytic class-activity evidence."""

    if not isinstance(probabilities, Sequence) or not probabilities:
        raise ThermalSCEncodingError("probabilities must be a non-empty sequence")
    cfg = _validate_config(config or ThermalSCEncodingConfig())
    records = tuple(
        encode_activity_balanced_probability(value, cfg, stream_index=index)
        for index, value in enumerate(probabilities)
    )
    label_values = tuple(range(len(records))) if labels is None else tuple(labels)
    if len(label_values) != len(records):
        raise ThermalSCEncodingError("labels must have the same length as probabilities")
    proxy = compute_class_activity_proxy(
        tuple((record.bitstream,) for record in records),
        label_values,
    )
    return ActivityBalancedEncodingBatch(
        records=records,
        summary=ActivityBalancedEncodingSummary(
            record_count=len(records),
            class_activity_proxy=proxy,
            total_dummy_streams_inserted=sum(
                record.dummy_streams_inserted for record in records
            ),
            max_dummy_streams_inserted=max(
                record.dummy_streams_inserted for record in records
            ),
            dummy_stream_overhead_ratio=sum(
                record.dummy_streams_inserted for record in records
            )
            / len(records),
        ),
    )


def _validate_config(config: ThermalSCEncodingConfig) -> ThermalSCEncodingConfig:
    if not isinstance(config, ThermalSCEncodingConfig):
        raise ThermalSCEncodingError("config must be a ThermalSCEncodingConfig")
    if not isinstance(config.bitstream_length, int) or config.bitstream_length < 2:
        raise ThermalSCEncodingError("bitstream_length must be an integer >= 2")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ThermalSCEncodingError("seed must be a non-negative integer")
    if not isinstance(config.rotation_stride, int) or config.rotation_stride <= 0:
        raise ThermalSCEncodingError("rotation_stride must be a positive integer")
    if (
        not isinstance(config.dummy_streams_per_record, int)
        or config.dummy_streams_per_record < 0
    ):
        raise ThermalSCEncodingError(
            "dummy_streams_per_record must be a non-negative integer"
        )
    if (
        isinstance(config.max_dummy_overhead_ratio, bool)
        or not isinstance(config.max_dummy_overhead_ratio, int | float)
        or not math.isfinite(float(config.max_dummy_overhead_ratio))
        or config.max_dummy_overhead_ratio < 0.0
    ):
        raise ThermalSCEncodingError(
            "max_dummy_overhead_ratio must be a finite non-negative value"
        )
    if config.dummy_streams_per_record > config.max_dummy_overhead_ratio:
        raise ThermalSCEncodingError(
            "dummy stream insertion exceeds max_dummy_overhead_ratio"
        )
    return config


def _validate_probability(probability: float) -> float:
    if isinstance(probability, bool) or not isinstance(probability, int | float):
        raise ThermalSCEncodingError("probability must be a finite value in [0, 1]")
    value = float(probability)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ThermalSCEncodingError("probability must be a finite value in [0, 1]")
    return value


def _distribute_ones(ones: int, bitstream_length: int) -> tuple[int, ...]:
    if ones == 0:
        return (0,) * bitstream_length
    if ones == bitstream_length:
        return (1,) * bitstream_length
    if ones > bitstream_length // 2:
        sparse_zero_stream = _distribute_ones(bitstream_length - ones, bitstream_length)
        return tuple(0 if bit else 1 for bit in sparse_zero_stream)

    stream = [0] * bitstream_length
    for rank in range(ones):
        position = math.floor((rank + 0.5) * bitstream_length / ones)
        stream[position] = 1
    return tuple(stream)


def _activity_preserving_rotation_offset(
    bitstream: tuple[int, ...],
    config: ThermalSCEncodingConfig,
    stream_index: int,
) -> int:
    desired = (config.seed + stream_index * config.rotation_stride) % config.bitstream_length
    target_transitions = compute_switching_activity((bitstream,)).total_transitions
    for distance in range(config.bitstream_length):
        for candidate in _candidate_offsets(desired, distance, config.bitstream_length):
            if (
                compute_switching_activity((_rotate(bitstream, candidate),)).total_transitions
                == target_transitions
            ):
                return candidate
    return 0


def _build_dummy_bitstreams(
    config: ThermalSCEncodingConfig,
    stream_index: int,
) -> tuple[tuple[int, ...], ...]:
    dummy_base = _distribute_ones(config.bitstream_length // 2, config.bitstream_length)
    return tuple(
        _rotate(
            dummy_base,
            _activity_preserving_rotation_offset(
                dummy_base,
                config,
                stream_index + dummy_index + 1,
            ),
        )
        for dummy_index in range(config.dummy_streams_per_record)
    )


def _candidate_offsets(desired: int, distance: int, bitstream_length: int) -> tuple[int, ...]:
    if distance == 0:
        return (desired,)
    return (
        (desired + distance) % bitstream_length,
        (desired - distance) % bitstream_length,
    )


def _rotate(bitstream: tuple[int, ...], offset: int) -> tuple[int, ...]:
    if offset == 0:
        return bitstream
    offset %= len(bitstream)
    return bitstream[offset:] + bitstream[:offset]
