# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio evidence chain verification

"""Every verdict the chain can reach, reached deliberately.

The acceptance cases behind these: tampered, missing, stale, wrong-model and
wrong-profile evidence must be rejected rather than exported under a digest that
makes them look checked. Each case here builds the pack that produces exactly
one of those verdicts, and the propagation cases prove a verdict travels the
whole length of a chain rather than one edge of it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from sc_neurocore.studio.evidence_chain import (
    CONTRADICTING_VERDICTS,
    EVIDENCE_CHAIN_SCHEMA_VERSION,
    EVIDENCE_VERIFIER_VERSION,
    EvidenceReceiptError,
    verify_evidence_chain,
)
from sc_neurocore.studio.evidence_receipt import (
    EVIDENCE_RECEIPT_KEY,
    EvidenceDependency,
    attach_evidence_receipt,
)

_EARLIER = datetime(2026, 9, 6, 10, 0, 0, tzinfo=timezone.utc)
_LATER = _EARLIER + timedelta(minutes=5)


def _training(job_id: str = "sj_train", *, at: datetime = _EARLIER) -> dict[str, Any]:
    """A training action's evidence, the root of the checkpoint chain."""
    return attach_evidence_receipt(
        {"action_kind": "studio.training", "loss": 0.25},
        lane="training",
        status="completed",
        binding="produced",
        scope={"action_kind": "studio.training", "job_id": job_id},
        now=at,
    )


def _restore(
    job_id: str = "sj_train",
    *,
    architecture: str = "64->128->10",
    at: datetime = _LATER,
) -> dict[str, Any]:
    """A materialised checkpoint, resting on the training job that wrote it."""
    return attach_evidence_receipt(
        {"restored": True, "source_job_id": job_id},
        lane="training",
        status="completed",
        binding="exported",
        scope={"architecture": architecture, "source_job_id": job_id},
        depends_on=(EvidenceDependency(lane="training", key="job_id", value=job_id),),
        now=at,
    )


def _verdicts(subjects: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {entry.name: entry.verdict for entry in verify_evidence_chain(subjects).entries}


class TestAVerifiedPack:
    def test_a_root_and_its_dependant_both_verify(self) -> None:
        assert _verdicts({"train": _training(), "restore": _restore()}) == {
            "train": "verified",
            "restore": "verified",
        }

    def test_a_verified_pack_is_verified_and_complete(self) -> None:
        report = verify_evidence_chain({"train": _training(), "restore": _restore()})

        assert report.verified is True
        assert report.complete is True
        assert report.unverified() == ()
        assert report.contradicted() == ()
        assert report.verdict_counts() == {"verified": 2}

    def test_the_document_names_the_verifier_that_reached_it(self) -> None:
        document = verify_evidence_chain({"train": _training()}).to_public_dict()

        assert document["schema_version"] == EVIDENCE_CHAIN_SCHEMA_VERSION
        assert document["verifier_version"] == EVIDENCE_VERIFIER_VERSION
        assert document["seal_algorithm"] == "sha256"
        verified_at = document["verified_at_utc"]
        assert isinstance(verified_at, str)
        assert verified_at.endswith("Z")
        entries = document["entries"]
        assert isinstance(entries, list)
        assert entries[0]["reason"] == "The payload matches its receipt."


class TestRejections:
    def test_an_edited_payload_is_tampered(self) -> None:
        payload = _training()
        payload["loss"] = 0.01

        verdict = _verdicts({"train": payload})

        assert verdict == {"train": "tampered"}

    def test_a_payload_that_can_no_longer_be_sealed_is_tampered(self) -> None:
        """An unsealable edit must not read as an absent seal."""
        payload = _training()
        payload["loss"] = float("inf")

        assert _verdicts({"train": payload}) == {"train": "tampered"}

    def test_an_input_outside_the_pack_is_reported_not_resolved(self) -> None:
        assert _verdicts({"restore": _restore()}) == {"restore": "missing_dependency"}

    def test_a_pack_missing_an_input_is_incomplete_but_not_contradicted(self) -> None:
        """Exporting a run without its inputs is incomplete, not dishonest."""
        report = verify_evidence_chain({"restore": _restore()})

        assert report.verified is True
        assert report.complete is False
        assert report.contradicted() == ()

    def test_disagreeing_with_an_input_about_the_model_is_a_scope_mismatch(self) -> None:
        """The wrong-model case: two subjects that cannot both be right."""
        training = attach_evidence_receipt(
            {"action_kind": "studio.training", "loss": 0.25},
            lane="training",
            status="completed",
            binding="produced",
            scope={"architecture": "64->32->10", "job_id": "sj_train"},
            now=_EARLIER,
        )

        assert _verdicts({"train": training, "restore": _restore()}) == {
            "train": "verified",
            "restore": "scope_mismatch",
        }

    def test_evidence_produced_before_its_input_is_stale(self) -> None:
        subjects = {"train": _training(at=_LATER), "restore": _restore(at=_EARLIER)}

        assert _verdicts(subjects) == {"train": "verified", "restore": "stale"}

    def test_a_payload_without_a_receipt_is_unsealed(self) -> None:
        assert _verdicts({"raw": {"loss": 0.25}}) == {"raw": "unsealed"}

    def test_an_unsealed_payload_does_not_contradict_the_pack(self) -> None:
        report = verify_evidence_chain({"raw": {"loss": 0.25}})

        assert report.verified is True
        assert report.complete is False

    def test_the_contradicting_verdicts_are_the_ones_that_refuse_an_export(self) -> None:
        assert set(CONTRADICTING_VERDICTS) == {"scope_mismatch", "stale", "tampered"}


class TestPropagation:
    def test_a_verdict_travels_the_whole_length_of_a_chain(self) -> None:
        """Three levels: an edit at the root must reach the grandchild."""
        training = _training()
        training["loss"] = 0.01
        attach = attach_evidence_receipt(
            {"attached": True, "source_job_id": "sj_train"},
            lane="training",
            status="completed",
            binding="exported",
            scope={"architecture": "64->128->10", "target_job_id": "sj_next"},
            depends_on=(
                EvidenceDependency(lane="training", key="source_job_id", value="sj_train"),
            ),
            now=_LATER + timedelta(minutes=5),
        )

        assert _verdicts({"train": training, "restore": _restore(), "attach": attach}) == {
            "train": "tampered",
            "restore": "stale",
            "attach": "stale",
        }

    def test_an_input_the_pack_could_not_resolve_does_not_make_its_child_stale(
        self,
    ) -> None:
        """Incompleteness must not turn into a refusal further down the chain."""
        attach = attach_evidence_receipt(
            {"attached": True, "source_job_id": "sj_train"},
            lane="training",
            status="completed",
            binding="exported",
            scope={"architecture": "64->128->10", "target_job_id": "sj_next"},
            depends_on=(
                EvidenceDependency(lane="training", key="source_job_id", value="sj_train"),
            ),
            now=_LATER + timedelta(minutes=5),
        )

        assert _verdicts({"restore": _restore(), "attach": attach}) == {
            "restore": "missing_dependency",
            "attach": "verified",
        }

    def test_a_pack_that_names_one_run_twice_still_resolves(self) -> None:
        """Two exports of one artefact must not leave the input unresolvable."""
        subjects = {
            "train": _training(),
            "train-again": _training(),
            "restore": _restore(),
        }

        assert _verdicts(subjects)["restore"] == "verified"


class TestMalformedPacks:
    def test_a_receipt_that_cannot_be_read_breaks_the_pack(self) -> None:
        payload = _training()
        payload[EVIDENCE_RECEIPT_KEY] = {**payload[EVIDENCE_RECEIPT_KEY], "lane": "invented"}

        with pytest.raises(EvidenceReceiptError):
            verify_evidence_chain({"train": payload})
