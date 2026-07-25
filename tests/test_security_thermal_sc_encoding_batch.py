# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Thermal SC batch mitigation contracts

"""Batch balancing, labels, and dummy-stream mitigation contracts."""

from .security_thermal_sc_encoding_support import *


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


@pytest.mark.parametrize("labels", [(0, True), (0, float("nan")), ("a", "b")])
def test_batch_encoder_rejects_non_finite_or_non_numeric_labels(labels) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    with pytest.raises(ThermalSCEncodingError, match="labels must be finite numeric values"):
        encode_activity_balanced_probabilities(
            (0.25, 0.5),
            ThermalSCEncodingConfig(bitstream_length=8),
            labels=labels,
        )
