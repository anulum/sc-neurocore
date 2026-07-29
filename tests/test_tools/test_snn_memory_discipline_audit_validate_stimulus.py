# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (validate_stimulus) from former test_snn_memory_discipline_audit.py

from __future__ import annotations

import os

from snn_memory_discipline_audit_support import *  # noqa: F403


def test_validate_stimulus_file_accepts_canonical_payload(tmp_path: Path) -> None:
    """A record with exactly the fleet schema has no violations."""

    path = tmp_path / "record.json"
    _write_json(path, _canonical_payload())

    violations = audit_tool.validate_stimulus_file(path, tmp_path, "SC-NEUROCORE")

    assert violations == ()


def test_validate_stimulus_file_accepts_recovery_grade_continuity_extension(
    tmp_path: Path,
) -> None:
    """The exact Tier-0 extension remains canonical and recovery-addressable."""

    path = tmp_path / "record.json"
    _write_json(
        path,
        _canonical_payload(
            kind="session_evidence",
            records={
                "session": ".coordination/sessions/SC-NEUROCORE/session.md",
                "handover": ".coordination/handovers/SC-NEUROCORE/handover.md",
            },
            seat="SC-NEUROCORE/codex-test",
            source_identity="SC-NEUROCORE/codex-test",
        ),
    )

    violations = audit_tool.validate_stimulus_file(path, tmp_path, "SC-NEUROCORE")

    assert violations == ()


def test_validate_stimulus_file_rejects_malformed_continuity_extension(
    tmp_path: Path,
) -> None:
    """Extension fields and kinds remain narrow instead of becoming aliases."""

    path = tmp_path / "record.json"
    _write_json(
        path,
        _canonical_payload(
            kind="event",
            records={"session": "elsewhere/session.txt"},
            seat="unscoped-seat",
            source_identity="other-project/codex-test",
        ),
    )

    violations = audit_tool.validate_stimulus_file(path, tmp_path, "SC-NEUROCORE")

    assert {item.code for item in violations} == {
        "invalid_kind",
        "invalid_records",
        "invalid_seat",
        "invalid_source_identity",
    }


def test_validate_stimulus_file_rejects_arbitrary_extra_key(tmp_path: Path) -> None:
    """An unrelated extension key cannot broaden the accepted schema."""

    path = tmp_path / "record.json"
    _write_json(path, _canonical_payload(uncontrolled_extension=True))

    violations = audit_tool.validate_stimulus_file(path, tmp_path, "SC-NEUROCORE")

    assert {item.code for item in violations} == {"noncanonical_keys"}


def test_validate_stimulus_file_rejects_legacy_aliases(tmp_path: Path) -> None:
    """Legacy summary/source-style records are reported as schema violations."""

    path = tmp_path / "record.json"
    _write_json(
        path,
        {
            "actor": "codex-seat-14753",
            "project": "SC-NEUROCORE",
            "summary": "legacy summary",
            "timestamp": "2026-07-09T161319Z",
            "unix_epoch": 1783613599,
        },
    )

    violations = audit_tool.validate_stimulus_file(path, tmp_path, "SC-NEUROCORE")

    assert {item.code for item in violations} >= {
        "noncanonical_keys",
        "invalid_content",
        "invalid_actor",
        "invalid_entities",
        "invalid_kind",
        "invalid_source_ref",
    }


def test_validate_stimulus_file_rejects_invalid_json_and_non_object(tmp_path: Path) -> None:
    """Malformed JSON and non-object JSON never pass the schema gate."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")

    assert audit_tool.validate_stimulus_file(bad_json, tmp_path, "SC-NEUROCORE") == (
        audit_tool.StimulusViolation(
            path="bad.json",
            code="invalid_json",
            detail="Expecting property name enclosed in double quotes at line 1, column 2",
        ),
    )
    assert audit_tool.validate_stimulus_file(list_json, tmp_path, "SC-NEUROCORE") == (
        audit_tool.StimulusViolation(
            path="list.json",
            code="invalid_payload",
            detail="top-level JSON value must be an object",
        ),
    )


def test_validate_stimulus_file_rejects_bad_timestamps(tmp_path: Path) -> None:
    """Boolean and unparsable timestamps are rejected."""

    boolean_path = tmp_path / "boolean.json"
    text_path = tmp_path / "text.json"
    _write_json(boolean_path, _canonical_payload(timestamp=False))
    _write_json(text_path, _canonical_payload(timestamp="not-a-date"))

    boolean_codes = {
        item.code
        for item in audit_tool.validate_stimulus_file(boolean_path, tmp_path, "SC-NEUROCORE")
    }
    text_codes = {
        item.code for item in audit_tool.validate_stimulus_file(text_path, tmp_path, "SC-NEUROCORE")
    }

    assert "invalid_timestamp" in boolean_codes
    assert "invalid_timestamp" in text_codes


def test_audit_accepts_later_canonical_append_only_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "legacy.json"
    successor = tmp_path / "successor.json"
    _write_json(legacy, {"summary": "Legacy noncanonical memory record."})
    _write_json(
        successor,
        _canonical_payload(
            content="Supersedes legacy.json. Canonical append-only remediation record."
        ),
    )
    os.utime(legacy, (1_000, 1_000))
    os.utime(successor, (2_000, 2_000))
    monkeypatch.setattr(
        audit_tool,
        "discover_snn_producers",
        lambda _repo: (
            audit_tool.ProducerCandidate(
                path="writer.py", function="write_stimulus", source_refs=("test",)
            ),
        ),
    )

    audit = audit_tool.audit_memory_discipline(tmp_path, tmp_path, "SC-NEUROCORE")

    assert audit.passed
    assert audit.checked_records == 2
    assert audit.violations == ()


def test_audit_rejects_noncanonical_or_older_supersession(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "legacy.json"
    older = tmp_path / "older.json"
    invalid = tmp_path / "invalid.json"
    _write_json(legacy, {"summary": "Legacy noncanonical memory record."})
    _write_json(
        older,
        _canonical_payload(content="Supersedes legacy.json. Canonical but older record."),
    )
    _write_json(
        invalid,
        _canonical_payload(
            actor="uncontrolled",
            content="Supersedes legacy.json. Later but noncanonical record.",
        ),
    )
    os.utime(older, (1_000, 1_000))
    os.utime(legacy, (2_000, 2_000))
    os.utime(invalid, (3_000, 3_000))
    monkeypatch.setattr(
        audit_tool,
        "discover_snn_producers",
        lambda _repo: (
            audit_tool.ProducerCandidate(
                path="writer.py", function="write_stimulus", source_refs=("test",)
            ),
        ),
    )

    audit = audit_tool.audit_memory_discipline(tmp_path, tmp_path, "SC-NEUROCORE")

    assert not audit.passed
    assert {violation.path for violation in audit.violations} == {"invalid.json", "legacy.json"}
