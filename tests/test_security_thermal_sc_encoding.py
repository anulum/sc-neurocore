# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import pytest

from sc_neurocore.security.side_channel_metrics import compute_class_activity_proxy
from sc_neurocore.security.thermal_sc_encoding import (
    ThermalSCEncodingConfig,
    ThermalSCEncodingError,
    _activity_preserving_rotation_offset,
    _distribute_ones,
    encode_activity_balanced_probability,
    encode_activity_balanced_probabilities,
)


def test_activity_balanced_encoder_preserves_probability_and_balances_switching() -> None:
    result = encode_activity_balanced_probability(
        0.5,
        ThermalSCEncodingConfig(bitstream_length=8, seed=11),
        stream_index=0,
    )

    assert result.bitstream.count(1) == 4
    assert result.realised_probability == 0.5
    assert result.activity_summary.per_stream_transition_counts == (7,)
    assert result.activity_summary.mean_transition_rate == 1.0
    assert result.evidence_boundary == "analytic_simulation_only"


def test_seed_domain_rotation_changes_phase_without_changing_activity_or_value() -> None:
    config = ThermalSCEncodingConfig(bitstream_length=16, seed=7, rotation_stride=3)

    first = encode_activity_balanced_probability(0.25, config, stream_index=0)
    second = encode_activity_balanced_probability(0.25, config, stream_index=1)

    assert first.bitstream != second.bitstream
    assert first.bitstream.count(1) == second.bitstream.count(1) == 4
    assert first.activity_summary == second.activity_summary
    assert first.seed_domain != second.seed_domain


def test_batch_encoder_reduces_class_activity_separation_against_unary_blocks() -> None:
    config = ThermalSCEncodingConfig(bitstream_length=16, seed=3)
    protected = encode_activity_balanced_probabilities((0.25, 0.5), config, labels=(10, 20))
    protected_samples = tuple((record.bitstream,) for record in protected.records)
    protected_proxy = compute_class_activity_proxy(protected_samples, (10, 20))

    baseline_samples = (
        (_correlated_activity_stream(0.25, 16),),
        (_correlated_activity_stream(0.5, 16),),
    )
    baseline_proxy = compute_class_activity_proxy(baseline_samples, (10, 20))

    assert protected_proxy.max_class_mean_gap < baseline_proxy.max_class_mean_gap
    assert protected.summary.class_activity_proxy == protected_proxy
    assert protected.summary.max_dummy_streams_inserted == 0
    assert protected.summary.dummy_stream_overhead_ratio == 0.0


def test_dummy_stream_insertion_is_explicit_and_budgeted() -> None:
    config = ThermalSCEncodingConfig(
        bitstream_length=8,
        seed=5,
        dummy_streams_per_record=2,
        max_dummy_overhead_ratio=2.0,
    )

    record = encode_activity_balanced_probability(0.25, config, stream_index=0)

    assert record.bitstream.count(1) == 2
    assert record.dummy_streams_inserted == 2
    assert len(record.dummy_bitstreams) == 2
    assert all(len(dummy) == 8 for dummy in record.dummy_bitstreams)
    assert record.activity_summary.stream_count == 3
    assert record.activity_summary.per_stream_transition_counts[0] == 4
    assert record.dummy_bitstreams[0] != record.dummy_bitstreams[1]


def test_batch_summary_accounts_for_dummy_stream_overhead() -> None:
    config = ThermalSCEncodingConfig(
        bitstream_length=8,
        seed=9,
        dummy_streams_per_record=1,
        max_dummy_overhead_ratio=1.0,
    )

    batch = encode_activity_balanced_probabilities((0.25, 0.5, 0.75), config)

    assert batch.summary.record_count == 3
    assert batch.summary.total_dummy_streams_inserted == 3
    assert batch.summary.max_dummy_streams_inserted == 1
    assert batch.summary.dummy_stream_overhead_ratio == 1.0


@pytest.mark.parametrize(
    ("probability", "config"),
    [
        (-0.1, ThermalSCEncodingConfig(bitstream_length=8)),
        (1.1, ThermalSCEncodingConfig(bitstream_length=8)),
        (0.5, ThermalSCEncodingConfig(bitstream_length=1)),
        (0.5, ThermalSCEncodingConfig(bitstream_length=8, rotation_stride=0)),
        (
            0.5,
            ThermalSCEncodingConfig(
                bitstream_length=8,
                dummy_streams_per_record=1,
                max_dummy_overhead_ratio=0.5,
            ),
        ),
    ],
)
def test_activity_balanced_encoder_rejects_invalid_contracts(
    probability: float,
    config: ThermalSCEncodingConfig,
) -> None:
    with pytest.raises(ThermalSCEncodingError):
        encode_activity_balanced_probability(probability, config)


def test_batch_encoder_rejects_empty_probability_list() -> None:
    with pytest.raises(ThermalSCEncodingError):
        encode_activity_balanced_probabilities((), ThermalSCEncodingConfig())


def test_batch_encoder_rejects_mismatched_labels() -> None:
    with pytest.raises(ThermalSCEncodingError):
        encode_activity_balanced_probabilities(
            (0.25, 0.5),
            ThermalSCEncodingConfig(),
            labels=(0,),
        )


def test_activity_balanced_encoder_rejects_boolean_stream_index() -> None:
    with pytest.raises(ThermalSCEncodingError, match="stream_index must be a non-negative integer"):
        encode_activity_balanced_probability(
            0.5,
            ThermalSCEncodingConfig(bitstream_length=8),
            stream_index=True,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("labels", [(0, True), (0, float("nan")), ("a", "b")])
def test_batch_encoder_rejects_non_finite_or_non_numeric_labels(labels) -> None:
    with pytest.raises(ThermalSCEncodingError, match="labels must be finite numeric values"):
        encode_activity_balanced_probabilities(
            (0.25, 0.5),
            ThermalSCEncodingConfig(bitstream_length=8),
            labels=labels,
        )


def test_validate_config_rejects_non_config_object() -> None:
    """A config of the wrong type is rejected before any field is read."""
    with pytest.raises(ThermalSCEncodingError, match="must be a ThermalSCEncodingConfig"):
        encode_activity_balanced_probability(0.5, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        ThermalSCEncodingConfig(bitstream_length=8, seed=-1),
        ThermalSCEncodingConfig(bitstream_length=8, dummy_streams_per_record=-1),
        ThermalSCEncodingConfig(bitstream_length=8, max_dummy_overhead_ratio=-1.0),
        ThermalSCEncodingConfig(bitstream_length=8, max_dummy_overhead_ratio=float("inf")),
    ],
)
def test_validate_config_rejects_out_of_range_fields(config: ThermalSCEncodingConfig) -> None:
    """A negative seed, negative dummy-stream count, and a negative or
    non-finite overhead ratio are each rejected as contract violations."""
    with pytest.raises(ThermalSCEncodingError):
        encode_activity_balanced_probability(0.5, config)


def test_validate_probability_rejects_non_numeric() -> None:
    """A non-numeric probability fails the type guard before the range check."""
    with pytest.raises(ThermalSCEncodingError, match=r"probability must be a finite value"):
        encode_activity_balanced_probability(
            "oops",  # type: ignore[arg-type]
            ThermalSCEncodingConfig(bitstream_length=8),
        )


@pytest.mark.parametrize(("probability", "expected_ones"), [(0.0, 0), (1.0, 8)])
def test_distribute_ones_handles_degenerate_densities(
    probability: float, expected_ones: int
) -> None:
    """All-zero and all-one densities take the dedicated fast paths instead of
    the interior rank-spreading loop."""
    result = encode_activity_balanced_probability(
        probability, ThermalSCEncodingConfig(bitstream_length=8)
    )
    assert result.bitstream.count(1) == expected_ones
    assert result.realised_probability == probability


def test_rotation_offset_falls_back_to_zero_without_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the candidate generator yields no offsets the activity-preserving
    search finds nothing and falls back to the safe identity rotation."""
    config = ThermalSCEncodingConfig(bitstream_length=8)
    base = _distribute_ones(4, config.bitstream_length)
    monkeypatch.setattr(
        "sc_neurocore.security.thermal_sc_encoding._candidate_offsets",
        lambda *args, **kwargs: (),
    )
    assert _activity_preserving_rotation_offset(base, config, 0) == 0


def _correlated_activity_stream(probability: float, bitstream_length: int) -> tuple[int, ...]:
    ones = round(probability * bitstream_length)
    if probability >= 0.5:
        return tuple(index % 2 for index in range(bitstream_length))
    return tuple(1 if index < ones else 0 for index in range(bitstream_length))
