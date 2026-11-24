# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (stream_validation) from former test_scnir_schema.py

"""Validation of SC-NIR stream fields, transforms, precision and random sources."""

from __future__ import annotations

import pytest
from sc_neurocore.ir.scnir_schema import (
    scnir_from_dict,
    scnir_to_dict,
    SCNIRDocument,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
    SCNIRStreamTransform,
    SCNIRValidationError,
    validate_scnir_dict,
)
from sc_neurocore.learning.online_o1 import (
    OnlineO1Config,
)
from tests.scnir_schema_support import (
    _valid_document,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bitstream_length", 0, "bitstream_length"),
        ("delay_steps", -1, "delay_steps"),
        ("delay_steps", [], "delay_steps"),
        ("delay_steps", [0, -1], "delay_steps"),
        ("encoding", "rate_only", "encoding"),
        ("signal_kind", "voltageish", "signal_kind"),
    ],
)
def test_scnir_rejects_invalid_stream_core_fields(field: str, value: object, message: str) -> None:
    """SC-NIR rejects invalid stream core fields."""
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0][field] = value

    with pytest.raises(SCNIRValidationError, match=message):
        validate_scnir_dict(payload)


def test_scnir_rejects_unknown_fields_fail_closed() -> None:
    """SC-NIR rejects unknown fields fail closed."""
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["unmodelled_runtime_hint"] = "accepting this would be unsafe"

    with pytest.raises(SCNIRValidationError, match="unknown"):
        validate_scnir_dict(payload)


def test_scnir_records_threshold_stream_transform_metadata() -> None:
    """SC-NIR records threshold stream transform metadata."""
    doc = SCNIRDocument(
        producer="sc-neurocore-test",
        streams=[
            SCNIRStream(
                stream_id="conn.li_to_lif.weight",
                layer="lif",
                bitstream_length=512,
                encoding="bipolar",
                signal_kind="weight",
                precision=SCNIRPrecision(
                    signed=True,
                    total_bits=16,
                    fractional_bits=8,
                    accumulator_bits=34,
                    rounding="nearest_even",
                    overflow="saturate",
                ),
                source=SCNIRSource(
                    kind="lfsr",
                    seed=11,
                    lfsr_polynomial="x^16 + x^14 + x^13 + x^11 + 1",
                    tap_mask=0xB400,
                ),
                transforms=[
                    SCNIRStreamTransform(
                        kind="threshold",
                        position="source",
                        comparison="greater_than",
                        values=(0.25, 0.5),
                    )
                ],
            )
        ],
    )

    payload = scnir_to_dict(doc)

    assert payload["streams"][0]["transforms"] == [
        {
            "kind": "threshold",
            "position": "source",
            "comparison": "greater_than",
            "values": [0.25, 0.5],
        }
    ]
    validate_scnir_dict(payload)


def test_scnir_rejects_invalid_threshold_transform_values() -> None:
    """SC-NIR rejects invalid threshold transform values."""
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["transforms"] = [
        {
            "kind": "threshold",
            "position": "source",
            "comparison": "greater_than",
            "values": [],
        }
    ]

    with pytest.raises(SCNIRValidationError, match="transforms"):
        validate_scnir_dict(payload)


def test_scnir_rejects_invalid_precision() -> None:
    """SC-NIR rejects invalid precision."""
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["precision"]["fractional_bits"] = 16

    with pytest.raises(SCNIRValidationError, match="fractional_bits"):
        validate_scnir_dict(payload)


def test_scnir_rejects_invalid_random_source_metadata() -> None:
    """SC-NIR rejects invalid random source metadata."""
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["source"].pop("lfsr_polynomial")

    with pytest.raises(SCNIRValidationError, match="lfsr_polynomial"):
        validate_scnir_dict(payload)


def test_scnir_rejects_invalid_correlation_reference() -> None:
    """SC-NIR rejects invalid correlation reference."""
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["correlation_constraints"][0]["peer_stream_id"] = "missing"

    with pytest.raises(SCNIRValidationError, match="peer_stream_id"):
        validate_scnir_dict(payload)


def test_scnir_records_validated_online_learning_metadata() -> None:
    """SC-NIR records validated online learning metadata."""
    annotation = OnlineO1Config(
        weight_bits=10,
        trace_bits=6,
        reward_bits=5,
        learning_shift=3,
        trace_decay_shift=2,
    ).to_scnir_annotation(rule_id="edge_reward_stdp")
    doc = _valid_document()
    stream = doc.streams[1]
    annotated = SCNIRDocument(
        producer=doc.producer,
        streams=(
            doc.streams[0],
            SCNIRStream(
                stream_id=stream.stream_id,
                layer=stream.layer,
                bitstream_length=stream.bitstream_length,
                encoding=stream.encoding,
                signal_kind=stream.signal_kind,
                precision=stream.precision,
                source=stream.source,
                online_learning=annotation,
            ),
        ),
    )

    payload = scnir_to_dict(annotated)

    assert payload["streams"][0]["online_learning"] is None
    assert payload["streams"][1]["online_learning"] == annotation
    assert scnir_to_dict(scnir_from_dict(payload)) == payload
    validate_scnir_dict(payload)


def test_scnir_rejects_online_learning_metadata_on_non_weight_stream() -> None:
    """SC-NIR rejects online learning metadata on non-weight stream."""
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["online_learning"] = OnlineO1Config().to_scnir_annotation(
        rule_id="bad_spike_rule"
    )

    with pytest.raises(SCNIRValidationError, match="online_learning"):
        validate_scnir_dict(payload)


def test_scnir_rejects_duplicate_stream_ids() -> None:
    """SC-NIR rejects duplicate stream ids."""
    payload = scnir_to_dict(_valid_document())
    payload["streams"][1]["stream_id"] = payload["streams"][0]["stream_id"]

    with pytest.raises(SCNIRValidationError, match="duplicate"):
        validate_scnir_dict(payload)
