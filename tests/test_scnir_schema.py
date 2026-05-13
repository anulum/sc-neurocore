# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR schema and validator

"""Contract tests for the stochastic-computing NIR metadata layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from sc_neurocore.cli import main
from sc_neurocore.ir.scnir_schema import (
    SCNIR_PREVIOUS_SCHEMA_VERSION,
    SCNIR_SCHEMA_VERSION,
    SCNIR_V02_SCHEMA_VERSION,
    SCNIRCorrelationConstraint,
    SCNIRDocument,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
    SCNIRStreamTransform,
    SCNIRValidationError,
    load_scnir,
    scnir_from_dict,
    scnir_to_dict,
    upgrade_scnir_dict,
    validate_scnir_dict,
    write_scnir,
)


def _valid_document() -> SCNIRDocument:
    return SCNIRDocument(
        producer="sc-neurocore-test",
        streams=[
            SCNIRStream(
                stream_id="layer0_input",
                layer="layer0",
                bitstream_length=1024,
                encoding="bipolar",
                signal_kind="spike",
                precision=SCNIRPrecision(
                    signed=True,
                    total_bits=16,
                    fractional_bits=8,
                    accumulator_bits=32,
                    rounding="nearest_even",
                    overflow="saturate",
                ),
                source=SCNIRSource(
                    kind="lfsr",
                    seed=17,
                    lfsr_polynomial="x^16 + x^14 + x^13 + x^11 + 1",
                    tap_mask=0xB400,
                ),
                correlation_constraints=[
                    SCNIRCorrelationConstraint(
                        peer_stream_id="layer0_weight",
                        policy="max_correlation",
                        max_abs_correlation=0.03,
                    )
                ],
            ),
            SCNIRStream(
                stream_id="layer0_weight",
                layer="layer0",
                bitstream_length=1024,
                encoding="unipolar",
                signal_kind="weight",
                precision=SCNIRPrecision(
                    signed=False,
                    total_bits=12,
                    fractional_bits=10,
                    accumulator_bits=24,
                    rounding="stochastic",
                    overflow="error",
                ),
                source=SCNIRSource(kind="sobol", sobol_dimension=3),
            ),
        ],
    )


def test_scnir_round_trip_is_deterministic(tmp_path: Path) -> None:
    doc = _valid_document()
    payload = scnir_to_dict(doc)

    assert payload["schema_version"] == SCNIR_SCHEMA_VERSION
    assert payload == scnir_to_dict(scnir_from_dict(payload))

    path = tmp_path / "model.scnir.json"
    write_scnir(path, doc)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert scnir_to_dict(load_scnir(path)) == payload


def test_scnir_upgrade_canonicalises_supported_current_payload() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["streams"] = list(reversed(payload["streams"]))

    upgraded = upgrade_scnir_dict(payload)

    assert upgraded == scnir_to_dict(scnir_from_dict(payload))
    validate_scnir_dict(upgraded)


def test_scnir_upgrade_migrates_v02_documents_with_signal_kind() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["schema_version"] = SCNIR_V02_SCHEMA_VERSION
    for stream in payload["streams"]:
        stream.pop("signal_kind")
        stream.pop("transforms")

    upgraded = upgrade_scnir_dict(payload)

    assert upgraded["schema_version"] == SCNIR_SCHEMA_VERSION
    streams = {stream["stream_id"]: stream for stream in upgraded["streams"]}
    assert streams["layer0_input"]["signal_kind"] == "spike"
    assert streams["layer0_weight"]["signal_kind"] == "weight"
    assert {tuple(stream["transforms"]) for stream in upgraded["streams"]} == {()}
    validate_scnir_dict(upgraded)


def test_scnir_upgrade_migrates_v03_documents_with_transform_metadata() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["schema_version"] = SCNIR_PREVIOUS_SCHEMA_VERSION
    for stream in payload["streams"]:
        stream.pop("transforms")

    upgraded = upgrade_scnir_dict(payload)

    assert upgraded["schema_version"] == SCNIR_SCHEMA_VERSION
    assert {tuple(stream["transforms"]) for stream in upgraded["streams"]} == {()}
    validate_scnir_dict(upgraded)


def test_scnir_upgrade_migrates_v01_documents_with_zero_delay_and_signal_kind() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["schema_version"] = "sc-neurocore.scnir.v0.1"
    for stream in payload["streams"]:
        stream.pop("delay_steps")
        stream.pop("signal_kind")
        stream.pop("transforms")

    upgraded = upgrade_scnir_dict(payload)

    assert upgraded["schema_version"] == SCNIR_SCHEMA_VERSION
    assert {stream["delay_steps"] for stream in upgraded["streams"]} == {0}
    assert {stream["signal_kind"] for stream in upgraded["streams"]} == {"spike", "weight"}
    assert {tuple(stream["transforms"]) for stream in upgraded["streams"]} == {()}
    validate_scnir_dict(upgraded)


def test_scnir_upgrade_rejects_unknown_schema_version() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["schema_version"] = "sc-neurocore.scnir.v9.9"

    with pytest.raises(SCNIRValidationError, match="unsupported SC-NIR schema_version"):
        upgrade_scnir_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bitstream_length", 0, "bitstream_length"),
        ("delay_steps", -1, "delay_steps"),
        ("encoding", "rate_only", "encoding"),
        ("signal_kind", "voltageish", "signal_kind"),
    ],
)
def test_scnir_rejects_invalid_stream_core_fields(field: str, value: object, message: str) -> None:
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0][field] = value

    with pytest.raises(SCNIRValidationError, match=message):
        validate_scnir_dict(payload)


def test_scnir_rejects_unknown_fields_fail_closed() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["unmodelled_runtime_hint"] = "accepting this would be unsafe"

    with pytest.raises(SCNIRValidationError, match="unknown"):
        validate_scnir_dict(payload)


def test_scnir_records_threshold_stream_transform_metadata() -> None:
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
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["precision"]["fractional_bits"] = 16

    with pytest.raises(SCNIRValidationError, match="fractional_bits"):
        validate_scnir_dict(payload)


def test_scnir_rejects_invalid_random_source_metadata() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["source"].pop("lfsr_polynomial")

    with pytest.raises(SCNIRValidationError, match="lfsr_polynomial"):
        validate_scnir_dict(payload)


def test_scnir_rejects_invalid_correlation_reference() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["correlation_constraints"][0]["peer_stream_id"] = "missing"

    with pytest.raises(SCNIRValidationError, match="peer_stream_id"):
        validate_scnir_dict(payload)


def test_scnir_rejects_duplicate_stream_ids() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["streams"][1]["stream_id"] = payload["streams"][0]["stream_id"]

    with pytest.raises(SCNIRValidationError, match="duplicate"):
        validate_scnir_dict(payload)


def test_scnir_json_schema_resource_is_bundled() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "schemas/scnir/scnir.schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    assert payload["$id"].endswith("/schemas/scnir/scnir.schema.json")
    assert payload["properties"]["schema_version"]["const"] == SCNIR_SCHEMA_VERSION
    assert "bitstream_length" in json.dumps(payload)
    assert "delay_steps" in json.dumps(payload)
    assert "signal_kind" in json.dumps(payload)
    assert "transforms" in json.dumps(payload)
    assert "correlation_constraints" in json.dumps(payload)


def test_scnir_validate_cli_reports_valid_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "valid.scnir.json"
    write_scnir(path, _valid_document())

    with mock.patch("sys.argv", ["sc-neurocore", "scnir", "validate", str(path)]):
        rc = main()

    assert rc == 0
    assert "SC-NIR valid" in capsys.readouterr().out


def test_scnir_validate_cli_reports_invalid_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.scnir.json"
    payload = scnir_to_dict(_valid_document())
    payload["streams"][0]["source"]["seed"] = -1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with mock.patch("sys.argv", ["sc-neurocore", "scnir", "validate", str(path)]):
        rc = main()

    assert rc == 1
    assert "seed" in capsys.readouterr().out


def test_scnir_upgrade_cli_writes_canonical_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.scnir.json"
    output_path = tmp_path / "upgraded.scnir.json"
    write_scnir(input_path, _valid_document())

    with mock.patch(
        "sys.argv",
        [
            "sc-neurocore",
            "scnir",
            "upgrade",
            str(input_path),
            "--output",
            str(output_path),
        ],
    ):
        rc = main()

    assert rc == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == scnir_to_dict(_valid_document())
    assert "SC-NIR upgraded" in capsys.readouterr().out
