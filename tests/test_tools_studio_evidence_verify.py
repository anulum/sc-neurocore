# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Offline Studio evidence pack verifier

"""A pack exported through the real routes, rechecked without the Studio.

The whole point of ST-13 is that a reader does not have to trust the exporter,
so these cases run the real HTTP export — including the JSON round trip through
the browser that used to make every recorded seal unreproducible — and then
verify the pack from disk with the offline tool. The failure cases edit the pack
after export, which is the only way tampering actually reaches a recipient.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from tools.studio_evidence_verify import (
    EvidencePackError,
    main,
    render_report,
    verify_pack,
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A Studio client whose job artefacts land in a directory this test owns."""
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path / "projects"))
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        job_default_timeout_seconds=30.0,
    )
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


def _problems(finding: dict[str, Any], label: str) -> list[str]:
    """Return the named problem list from a verification finding."""
    problems = finding[f"{label}_problems"]
    assert isinstance(problems, list)
    return [str(problem) for problem in problems]


def _through_the_browser(payload: Any) -> Any:
    """Return a payload as the operator's browser hands it back."""
    return json.loads(json.dumps(payload))


def _pack_directory(tmp_path: Path) -> Path:
    """Return the directory holding the exported pack."""
    manifests = sorted(tmp_path.glob("**/evidence/manifest.json"))
    assert len(manifests) == 1, manifests
    return manifests[0].parent.parent


def _export(client: TestClient, tmp_path: Path) -> Path:
    """Run one simulation, export it as a pack, and return the pack directory."""
    simulation = client.post(
        "/api/models/simulate",
        json={"name": "AdExNeuron", "duration": 20.0, "current": 50.0, "dt": 0.1},
    )
    assert simulation.status_code == 200, simulation.text
    export = client.post(
        "/api/studio/evidence/bundle",
        json={
            "include_audit": False,
            "simulation_results": [_through_the_browser(simulation.json())],
        },
    )
    assert export.status_code == 200, export.text
    return _pack_directory(tmp_path)


class TestAnExportedPack:
    def test_a_pack_exported_through_the_browser_verifies_offline(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The acceptance case: the seal survives the round trip and rechecks."""
        pack = _export(client, tmp_path)

        finding = verify_pack(pack)

        assert finding["verified"] is True
        assert finding["digest_problems"] == []
        assert finding["agreement_problems"] == []
        subject_count = finding["subject_count"]
        assert isinstance(subject_count, int)
        assert subject_count >= 1
        chain = finding["chain"]
        assert isinstance(chain, dict)
        assert chain["verdict_counts"] == {"verified": subject_count}

    def test_the_receipt_the_run_produced_survives_into_the_pack(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """An export must not replace an attestation of what ran."""
        pack = _export(client, tmp_path)

        document = json.loads((pack / "evidence/simulations/000.json").read_text(encoding="utf-8"))

        assert document["evidence_receipt"]["binding"] == "produced"
        assert document["evidence_receipt"]["lane"] == "simulation"
        assert document["evidence_receipt"]["scope"]["model_class"] == "AdExNeuron"

    def test_the_command_line_reports_and_succeeds(
        self, client: TestClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pack = _export(client, tmp_path)

        assert main([str(pack)]) == 0

        printed = capsys.readouterr().out
        assert "result: verified" in printed
        assert "verdicts: verified" in printed

    def test_the_finding_can_be_written_as_json(self, client: TestClient, tmp_path: Path) -> None:
        pack = _export(client, tmp_path)
        destination = tmp_path / "finding.json"

        assert main([str(pack), "--json", str(destination)]) == 0

        document = json.loads(destination.read_text(encoding="utf-8"))
        assert document["verified"] is True
        assert destination.read_text(encoding="utf-8").endswith("\n")


class TestTamperingAfterExport:
    def test_an_edited_payload_is_caught_by_its_digest_and_its_seal(
        self, client: TestClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pack = _export(client, tmp_path)
        target = pack / "evidence/simulations/000.json"
        document = json.loads(target.read_text(encoding="utf-8"))
        document["spike_count"] = document["spike_count"] + 1
        target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        assert main([str(pack)]) == 1

        captured = capsys.readouterr()
        assert "tampered" in captured.out
        assert "did not verify" in captured.err

    def test_a_file_removed_after_export_is_named(self, client: TestClient, tmp_path: Path) -> None:
        pack = _export(client, tmp_path)
        (pack / "evidence/simulations/000.json").unlink()

        finding = verify_pack(pack)

        assert finding["verified"] is False
        assert any("absent from the pack" in problem for problem in _problems(finding, "digest"))

    def test_a_recorded_verdict_that_does_not_match_is_named(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The exporter wrote both the evidence and its verdict."""
        pack = _export(client, tmp_path)
        chain_path = pack / "evidence/chain.json"
        document = json.loads(chain_path.read_text(encoding="utf-8"))
        document["entries"] = []
        chain_path.write_text(json.dumps(document), encoding="utf-8")

        finding = verify_pack(pack)

        assert finding["verified"] is False
        assert any(
            "the recorded chain does not list" in problem
            for problem in _problems(finding, "agreement")
        )

    def test_a_pack_without_a_chain_document_is_not_taken_on_trust(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        pack = _export(client, tmp_path)
        (pack / "evidence/chain.json").unlink()

        finding = verify_pack(pack)

        assert finding["verified"] is False
        assert finding["agreement_problems"] == [
            "evidence/chain.json is absent; the pack records no verdict of its own."
        ]


def _hand_built_pack(tmp_path: Path, files: dict[str, str], entries: list[Any]) -> Path:
    """Write a pack whose manifest names exactly ``entries``."""
    for relative_path, text in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    manifest = tmp_path / "evidence" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"bundle_id": "seb_hand", "entries": entries}), encoding="utf-8")
    return tmp_path


def _entry(pack: Path, relative_path: str, entry_type: str = "job_artifact") -> dict[str, Any]:
    """Return a manifest entry describing a file already written."""
    payload = (pack / relative_path).read_bytes()
    return {
        "bundle_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "type": entry_type,
    }


class TestFilesThePackCannotCheck:
    def test_a_manifest_entry_naming_no_file_is_reported(self, tmp_path: Path) -> None:
        pack = _hand_built_pack(tmp_path, {}, [{"type": "job_record", "sha256": "0" * 64}])

        finding = verify_pack(pack)

        assert finding["digest_problems"] == [
            "A manifest entry of type 'job_record' names no file."
        ]

    def test_a_file_of_the_wrong_size_is_reported(self, tmp_path: Path) -> None:
        pack = _hand_built_pack(tmp_path, {"evidence/notes.txt": "hello\n"}, [])
        entry = _entry(pack, "evidence/notes.txt")
        entry["size_bytes"] = 1
        (pack / "evidence" / "manifest.json").write_text(
            json.dumps({"bundle_id": "seb_hand", "entries": [entry]}), encoding="utf-8"
        )

        finding = verify_pack(pack)

        assert any(
            "is 6 bytes, not the recorded 1" in problem for problem in _problems(finding, "digest")
        )

    @pytest.mark.parametrize(
        ("name", "text"),
        [
            ("evidence/notes.txt", "not json at all"),
            ("evidence/list.json", "[1, 2, 3]"),
            ("evidence/plain.json", '{"result": 1}'),
        ],
    )
    def test_a_file_without_a_readable_receipt_is_not_a_subject(
        self, tmp_path: Path, name: str, text: str
    ) -> None:
        """Skipped, not failed: a pack may carry files that are not evidence."""
        pack = _hand_built_pack(tmp_path, {name: text}, [])
        (pack / "evidence" / "manifest.json").write_text(
            json.dumps({"bundle_id": "seb_hand", "entries": [_entry(pack, name)]}),
            encoding="utf-8",
        )

        finding = verify_pack(pack)

        assert finding["subject_count"] == 0
        assert finding["digest_problems"] == []


class TestARecordedChainThatCannotBeRead:
    @pytest.mark.parametrize(
        "entries",
        ["not a list", [7], [{"name": 1, "verdict": "verified"}]],
    )
    def test_an_unreadable_recorded_entry_records_no_verdict(
        self, tmp_path: Path, entries: Any
    ) -> None:
        """A chain nobody can read must not be treated as agreement."""
        pack = _hand_built_pack(tmp_path, {}, [])
        (pack / "evidence" / "chain.json").write_text(
            json.dumps({"entries": entries}), encoding="utf-8"
        )

        finding = verify_pack(pack)

        assert finding["agreement_problems"] == []
        assert finding["verified"] is True


class TestUnreadablePacks:
    def test_a_directory_that_is_not_a_pack_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(EvidencePackError, match="not an evidence pack"):
            verify_pack(tmp_path)

    def test_the_command_line_reports_an_unreadable_pack(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([str(tmp_path)]) == 1

        assert "cannot be verified" in capsys.readouterr().err

    def test_a_manifest_that_is_not_an_object_is_refused(self, tmp_path: Path) -> None:
        manifest = tmp_path / "evidence" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("[]", encoding="utf-8")

        with pytest.raises(EvidencePackError, match="does not hold a JSON object"):
            verify_pack(tmp_path)

    def test_a_manifest_without_entries_is_refused(self, tmp_path: Path) -> None:
        manifest = tmp_path / "evidence" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"bundle_id": "seb_x"}', encoding="utf-8")

        with pytest.raises(EvidencePackError, match="must list its entries"):
            verify_pack(tmp_path)

    def test_a_chain_document_that_is_not_an_object_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "evidence").mkdir(parents=True)
        (tmp_path / "evidence" / "manifest.json").write_text(
            '{"bundle_id": "seb_x", "entries": []}', encoding="utf-8"
        )
        (tmp_path / "evidence" / "chain.json").write_text("[]", encoding="utf-8")

        with pytest.raises(EvidencePackError, match="does not hold a JSON object"):
            verify_pack(tmp_path)


class TestReportRendering:
    def _finding(self, **overrides: Any) -> dict[str, Any]:
        finding: dict[str, Any] = {
            "agreement_problems": [],
            "bundle_id": "seb_x",
            "chain": {
                "entries": [],
                "verdict_counts": {},
                "verified": True,
                "verifier_version": "v",
            },
            "digest_problems": [],
            "subject_count": 0,
            "verified": True,
            "verifier_version": "v",
        }
        finding.update(overrides)
        return finding

    def test_an_empty_pack_reports_no_verdicts(self) -> None:
        report = render_report(self._finding())

        assert "verdicts: none" in report
        assert "result: verified" in report

    def test_problems_and_failures_are_both_named(self) -> None:
        report = render_report(
            self._finding(
                agreement_problems=["a mismatch"],
                digest_problems=["a bad digest"],
                verified=False,
                chain={
                    "entries": [
                        {
                            "name": "evidence/simulations/000.json",
                            "reason": "it changed",
                            "receipt_id": "simulation.abc",
                            "verdict": "tampered",
                        }
                    ],
                    "verdict_counts": {"tampered": 1},
                    "verified": False,
                    "verifier_version": "v",
                },
            )
        )

        assert "digest problems:" in report
        assert "agreement problems:" in report
        assert "tampered: evidence/simulations/000.json — it changed" in report
        assert "result: NOT VERIFIED" in report

    def test_a_finding_without_a_chain_mapping_is_refused(self) -> None:
        with pytest.raises(TypeError, match="expected a mapping"):
            render_report(self._finding(chain=[]))

    def test_a_chain_whose_entries_are_not_a_list_lists_no_failures(self) -> None:
        report = render_report(
            self._finding(
                chain={
                    "entries": "unreadable",
                    "verdict_counts": {"verified": 1},
                    "verified": True,
                    "verifier_version": "v",
                }
            )
        )

        assert "subjects that did not verify" not in report
