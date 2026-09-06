# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio evidence receipts

"""What a receipt says, and what it refuses to say.

A receipt is the only thing standing between an exported artefact and a reader
who has to take the exporter's word for it. These cases hold its shape: it seals
the subject and not itself, it names what it rests on by value rather than by
position, it distinguishes attesting production from attesting an export, and a
receipt that cannot be read is an error rather than an absent one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from sc_neurocore.studio.evidence_receipt import (
    EVIDENCE_BINDINGS,
    EVIDENCE_RECEIPT_KEY,
    EVIDENCE_RECEIPT_SCHEMA_VERSION,
    EvidenceDependency,
    EvidenceReceiptError,
    attach_evidence_receipt,
    build_evidence_receipt,
    is_sha256_hex,
    parse_timestamp,
    read_evidence_receipt,
    subject_of,
    utc_timestamp,
    validate_binding,
    validate_lane,
    validate_status,
)
from sc_neurocore.studio.evidence_seal import EvidenceSealError, seal_sha256

_MOMENT = datetime(2026, 9, 6, 11, 22, 33, tzinfo=timezone.utc)
_PAYLOAD: dict[str, Any] = {"spike_count": 3.0, "states": {"v": [0.1, -70.0]}}


def _sealed(**overrides: Any) -> dict[str, Any]:
    """Return a sealed simulation payload with the given receipt overrides."""
    fields: dict[str, Any] = {
        "lane": "simulation",
        "status": "completed",
        "binding": "produced",
        "scope": {"experiment_sha256": "a" * 64},
        "now": _MOMENT,
    }
    fields.update(overrides)
    return attach_evidence_receipt(_PAYLOAD, **fields)


class TestSealingTheSubject:
    def test_the_receipt_seals_the_payload_without_itself(self) -> None:
        payload = _sealed()
        receipt = read_evidence_receipt(payload)

        assert receipt is not None
        assert receipt.seal_sha256 == seal_sha256(_PAYLOAD)
        assert receipt.seal_sha256 == seal_sha256(subject_of(payload))

    def test_re_sealing_an_exported_payload_reproduces_the_receipt(self) -> None:
        """A payload that already carries one must not seal its own receipt."""
        once = _sealed()
        twice = attach_evidence_receipt(
            once,
            lane="simulation",
            status="completed",
            binding="produced",
            scope={"experiment_sha256": "a" * 64},
            now=_MOMENT,
        )

        assert twice == once

    def test_the_identifier_is_the_seal_so_one_artefact_has_one_identity(self) -> None:
        receipt = read_evidence_receipt(_sealed())

        assert receipt is not None
        assert receipt.receipt_id == f"simulation.{receipt.seal_sha256[:32]}"

    def test_a_changed_payload_gets_a_different_identifier(self) -> None:
        other = attach_evidence_receipt(
            {**_PAYLOAD, "spike_count": 4.0},
            lane="simulation",
            status="completed",
            binding="produced",
            scope={"experiment_sha256": "a" * 64},
            now=_MOMENT,
        )
        first = read_evidence_receipt(_sealed())
        second = read_evidence_receipt(other)

        assert first is not None and second is not None
        assert first.receipt_id != second.receipt_id

    def test_a_payload_that_cannot_be_sealed_is_refused(self) -> None:
        with pytest.raises(EvidenceSealError):
            attach_evidence_receipt(
                {"loss": float("nan")},
                lane="training",
                status="completed",
                binding="produced",
                scope={},
            )


class TestWhatTheReceiptRecords:
    def test_the_public_block_carries_the_whole_contract(self) -> None:
        payload = _sealed(
            depends_on=(EvidenceDependency(lane="training", key="job_id", value="sj_1"),)
        )
        block = payload[EVIDENCE_RECEIPT_KEY]

        assert block["schema_version"] == EVIDENCE_RECEIPT_SCHEMA_VERSION
        assert block["lane"] == "simulation"
        assert block["status"] == "completed"
        assert block["binding"] == "produced"
        assert block["seal_algorithm"] == "sha256"
        assert block["produced_at_utc"] == "2026-09-06T11:22:33Z"
        assert block["depends_on"] == [{"key": "job_id", "lane": "training", "value": "sj_1"}]

    def test_the_scope_is_ordered_so_two_exports_agree_byte_for_byte(self) -> None:
        payload = _sealed(scope={"z": "1", "a": "2"})
        block = payload[EVIDENCE_RECEIPT_KEY]

        assert list(block["scope"]) == ["a", "z"]

    def test_the_binding_tells_production_from_export(self) -> None:
        produced = read_evidence_receipt(_sealed(binding="produced"))
        exported = read_evidence_receipt(_sealed(binding="exported"))

        assert produced is not None and exported is not None
        assert produced.binding == "produced"
        assert exported.binding == "exported"
        assert set(EVIDENCE_BINDINGS) == {"exported", "produced"}

    def test_a_timestamp_is_recorded_to_the_second_in_utc(self) -> None:
        assert utc_timestamp(_MOMENT) == "2026-09-06T11:22:33Z"
        assert utc_timestamp().endswith("Z")
        assert parse_timestamp("2026-09-06T11:22:33Z") == _MOMENT


class TestReading:
    def test_a_payload_without_a_receipt_reads_as_none(self) -> None:
        assert read_evidence_receipt(_PAYLOAD) is None

    def test_a_read_receipt_round_trips_through_its_public_form(self) -> None:
        payload = _sealed(
            depends_on=(EvidenceDependency(lane="training", key="job_id", value="sj_1"),)
        )
        receipt = read_evidence_receipt(payload)

        assert receipt is not None
        assert receipt.to_public_dict() == payload[EVIDENCE_RECEIPT_KEY]

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ({"schema_version": "studio.evidence-receipt.v99"}, "not the contract"),
            ({"seal_sha256": "short"}, "requires a SHA-256 seal"),
            ({"seal_algorithm": "md5"}, "sha256 seal algorithm"),
            ({"receipt_id": ""}, "requires an identifier"),
            ({"produced_at_utc": 17}, "requires a production timestamp"),
            ({"produced_at_utc": "2026-09-06T11:22:33"}, "must be UTC"),
            ({"produced_at_utc": "not a time Z"}, "ISO-8601"),
            ({"lane": "invented"}, "not a Studio evidence class"),
            ({"lane": 4}, "requires an evidence class"),
            ({"status": "half-done"}, "not a terminal evidence status"),
            ({"status": None}, "requires a terminal status"),
            ({"binding": "asserted"}, "not an evidence receipt binding"),
            ({"scope": []}, "scope must be an object"),
            ({"scope": {"a": 1}}, "map names to text"),
            ({"depends_on": {}}, "dependencies must be a list"),
            ({"depends_on": ["x"]}, "dependency must be an object"),
            ({"depends_on": [{"lane": "training", "key": "", "value": "v"}]}, "requires a key"),
        ],
    )
    def test_a_malformed_receipt_is_an_error_not_an_absent_one(
        self, mutation: dict[str, Any], message: str
    ) -> None:
        payload = _sealed()
        payload[EVIDENCE_RECEIPT_KEY] = {**payload[EVIDENCE_RECEIPT_KEY], **mutation}

        with pytest.raises(EvidenceReceiptError, match=message):
            read_evidence_receipt(payload)

    def test_a_receipt_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(EvidenceReceiptError, match="must be an object"):
            read_evidence_receipt({EVIDENCE_RECEIPT_KEY: "sealed, honest"})


class TestValidators:
    def test_a_lane_and_status_and_binding_are_returned_when_controlled(self) -> None:
        assert validate_lane("training") == "training"
        assert validate_status("cancelled") == "cancelled"
        assert validate_binding("exported") == "exported"

    def test_a_digest_is_recognised_only_in_lowercase_hexadecimal(self) -> None:
        assert is_sha256_hex("a" * 64)
        assert not is_sha256_hex("A" * 64)
        assert not is_sha256_hex("a" * 63)
        assert not is_sha256_hex(64)

    @pytest.mark.parametrize(
        "override",
        [{"lane": "invented"}, {"status": "half"}, {"binding": "claimed"}],
    )
    def test_building_refuses_an_uncontrolled_field_before_sealing(
        self, override: dict[str, str]
    ) -> None:
        fields: dict[str, Any] = {
            "lane": "simulation",
            "status": "completed",
            "binding": "produced",
            "scope": {},
            "produced_at_utc": "2026-09-06T11:22:33Z",
        }
        fields.update(override)

        with pytest.raises(EvidenceReceiptError):
            build_evidence_receipt(_PAYLOAD, **fields)
