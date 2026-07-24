# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (validate_stimulus) from former test_snn_memory_discipline_audit.py

from __future__ import annotations

from snn_memory_discipline_audit_support import *  # noqa: F403


def test_validate_stimulus_file_accepts_canonical_payload(tmp_path: Path) -> None:
    """A record with exactly the fleet schema has no violations."""

    path = tmp_path / "record.json"
    _write_json(path, _canonical_payload())

    violations = audit_tool.validate_stimulus_file(path, tmp_path, "SC-NEUROCORE")

    assert violations == ()


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
