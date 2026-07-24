# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (round_trip_and_upgrade) from former test_scnir_schema.py

from __future__ import annotations

from tests.scnir_schema_support import *  # noqa: F403


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


def test_scnir_preserves_delay_step_vectors() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["streams"][1]["delay_steps"] = [1, 2]

    validate_scnir_dict(payload)
    round_tripped = scnir_to_dict(scnir_from_dict(payload))

    assert round_tripped["streams"][1]["delay_steps"] == [1, 2]


def test_scnir_upgrade_migrates_v05_documents_with_empty_hierarchy() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["schema_version"] = SCNIR_PREVIOUS_SCHEMA_VERSION
    payload.pop("hierarchy")

    upgraded = upgrade_scnir_dict(payload)

    assert upgraded["schema_version"] == SCNIR_SCHEMA_VERSION
    assert upgraded["hierarchy"] == []
    validate_scnir_dict(upgraded)


def test_scnir_upgrade_rejects_unknown_schema_version() -> None:
    payload = scnir_to_dict(_valid_document())
    payload["schema_version"] = "sc-neurocore.scnir.v9.9"

    with pytest.raises(SCNIRValidationError, match="unsupported SC-NIR schema_version"):
        upgrade_scnir_dict(payload)
