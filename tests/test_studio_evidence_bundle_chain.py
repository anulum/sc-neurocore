# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio evidence bundle chain contracts

"""What the exporter seals, what it carries through, and what it refuses.

A bundle used to be an unconditional copy: whatever the operator handed over was
written out under a digest of itself. These cases hold the three things that
changed — an unsealed payload is sealed and marked as attesting only the export,
a payload that arrives attesting its own production keeps that attestation, and
a pack whose parts contradict each other is refused rather than published.
"""

from __future__ import annotations

from tests.studio_evidence_bundle_support import *  # noqa: F403

from sc_neurocore.studio.evidence_chain import EVIDENCE_VERIFIER_VERSION
from sc_neurocore.studio.evidence_receipt import (
    EVIDENCE_RECEIPT_KEY,
    attach_evidence_receipt,
    read_evidence_receipt,
)


def _context(tmp_path: Path) -> StudioJobContext:
    """Return a job context whose artefacts land under ``tmp_path``."""
    return StudioJobContext(
        job_id="sj_evidence",
        work_dir=tmp_path / "evidence",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )


def _bundle_document(tmp_path: Path, relative_path: str) -> dict[str, object]:
    """Read one written bundle file back from disk."""
    payload = json.loads((tmp_path / "evidence" / relative_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


class TestSealingWhatArrivesUnsealed:
    def test_an_unsealed_payload_is_sealed_and_marked_as_an_export(self, tmp_path: Path) -> None:
        """An export-time receipt attests what was handed over, nothing more."""
        write_studio_evidence_bundle(
            _context(tmp_path), simulation_payloads=(_simulation_payload(),)
        )

        document = _bundle_document(tmp_path, "evidence/simulations/000.json")
        receipt = read_evidence_receipt(document)

        assert receipt is not None
        assert receipt.binding == "exported"
        assert receipt.lane == "simulation"

    def test_a_payload_that_attests_its_own_production_keeps_that_attestation(
        self, tmp_path: Path
    ) -> None:
        produced = attach_evidence_receipt(
            _simulation_payload(),
            lane="simulation",
            status="completed",
            binding="produced",
            scope={"experiment_sha256": "e" * 64},
        )

        write_studio_evidence_bundle(_context(tmp_path), simulation_payloads=(produced,))

        document = _bundle_document(tmp_path, "evidence/simulations/000.json")

        assert document[EVIDENCE_RECEIPT_KEY] == produced[EVIDENCE_RECEIPT_KEY]

    def test_every_classified_entry_names_its_receipt_in_the_manifest(self, tmp_path: Path) -> None:
        result = write_studio_evidence_bundle(
            _context(tmp_path),
            project_payload=_project_payload(),
            simulation_payloads=(_simulation_payload(),),
            analysis_payloads=(_analysis_payload(),),
        )

        entries = cast(list[dict[str, object]], result.manifest["entries"])
        classified = [entry for entry in entries if "evidence_classification" in entry]

        assert len(classified) == 3
        assert all(str(entry["receipt_id"]) for entry in classified)


class TestTheChainDocument:
    def test_the_pack_carries_its_own_verdict(self, tmp_path: Path) -> None:
        result = write_studio_evidence_bundle(
            _context(tmp_path), simulation_payloads=(_simulation_payload(),)
        )

        chain = _bundle_document(tmp_path, "evidence/chain.json")

        assert chain["schema_version"] == "studio.evidence-chain.v1"
        assert chain["verifier_version"] == EVIDENCE_VERIFIER_VERSION
        assert chain["verified"] is True
        assert chain["complete"] is True
        entries = cast(list[dict[str, object]], chain["entries"])
        assert [entry["name"] for entry in entries] == ["evidence/simulations/000.json"]
        summary = cast(dict[str, object], result.summary)
        assert summary["chain_verified"] is True
        assert summary["chain_verdict_counts"] == {"verified": 1}

    def test_an_input_outside_the_pack_is_reported_without_refusing_the_export(
        self, tmp_path: Path
    ) -> None:
        """A restore exported without its training job is incomplete, not false."""
        write_studio_evidence_bundle(
            _context(tmp_path), weight_restore_payloads=(_weight_restore_response(),)
        )

        chain = _bundle_document(tmp_path, "evidence/chain.json")
        entries = cast(list[dict[str, object]], chain["entries"])

        assert entries[0]["verdict"] == "missing_dependency"
        assert chain["verified"] is True
        assert chain["complete"] is False


class TestRefusingAPackThatContradictsItself:
    def test_attaching_a_checkpoint_of_another_shape_is_refused(self, tmp_path: Path) -> None:
        """The wrong-model case for the training lane, caught at export."""
        attach = dict(_weight_restore_attach_response())
        attach["target_architecture"] = "64->999->10"

        with pytest.raises(ValueError, match="contradicts itself"):
            write_studio_evidence_bundle(
                _context(tmp_path),
                weight_restore_payloads=(_weight_restore_response(),),
                weight_restore_attach_payloads=(attach,),
            )

    def test_an_agreeing_attach_and_restore_export_together(self, tmp_path: Path) -> None:
        result = write_studio_evidence_bundle(
            _context(tmp_path),
            weight_restore_payloads=(_weight_restore_response(),),
            weight_restore_attach_payloads=(_weight_restore_attach_response(),),
        )

        summary = cast(dict[str, object], result.summary)

        assert summary["chain_verified"] is True

    def test_a_payload_sealed_against_different_bytes_is_refused(self, tmp_path: Path) -> None:
        """A receipt copied onto another payload must not survive the export."""
        produced = attach_evidence_receipt(
            _simulation_payload(),
            lane="simulation",
            status="completed",
            binding="produced",
            scope={"experiment_sha256": "e" * 64},
        )
        produced["spike_count"] = 99

        with pytest.raises(ValueError, match="tampered"):
            write_studio_evidence_bundle(_context(tmp_path), simulation_payloads=(produced,))
