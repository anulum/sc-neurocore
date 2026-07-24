# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (repair_stimulus) from former test_snn_memory_discipline_audit.py

from __future__ import annotations

from snn_memory_discipline_audit_support import *  # noqa: F403


def test_repair_stimulus_file_normalises_legacy_payload(tmp_path: Path) -> None:
    """The repair path preserves legacy facts in canonical fields."""

    path = tmp_path / "legacy.json"
    _write_json(
        path,
        {
            "actor": "codex-seat-14753",
            "commit": "abc123",
            "evidence": ["focused tests passed", "strict mypy passed"],
            "kind": "todo_closure",
            "project": "SC-NEUROCORE",
            "summary": "Closed a SC-NEUROCORE TODO item.",
            "timestamp": "2026-07-09T161319Z",
            "todo_rows_closed": ["AUDIT-1"],
            "unix_epoch": 1783613599,
        },
    )

    repaired = audit_tool.repair_stimulus_file(path, "SC-NEUROCORE")
    violations = audit_tool.validate_stimulus_file(path, tmp_path, "SC-NEUROCORE")

    assert violations == ()
    assert set(repaired) == audit_tool.CANONICAL_KEYS
    assert repaired["actor"] == "codex"
    assert repaired["kind"] == "event"
    assert repaired["source_ref"] == "abc123"
    assert repaired["entities"] == ["SC-NEUROCORE"]
    assert "focused tests passed" in str(repaired["content"])
    assert "todo_rows_closed" not in json.loads(path.read_text(encoding="utf-8"))


def test_repair_stimulus_file_rejects_non_object_payload(tmp_path: Path) -> None:
    """Repair mode fails loudly when a record is not a JSON object."""

    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level JSON value must be an object"):
        audit_tool.repair_stimulus_file(path, "SC-NEUROCORE")


def test_repair_uses_unix_epoch_when_timestamp_is_missing(tmp_path: Path) -> None:
    """Legacy unix_epoch values remain valid canonical timestamps."""

    path = tmp_path / "legacy.json"
    _write_json(
        path,
        {
            "actor": "operator",
            "content": "Legacy record with timestamp fallback.",
            "project": "SC-NEUROCORE",
            "unix_epoch": 1783613599,
        },
    )

    repaired = audit_tool.repair_stimulus_file(path, "SC-NEUROCORE")

    assert repaired["timestamp"] == 1783613599


def test_repair_preserves_valid_kind_entities_and_source_ref(tmp_path: Path) -> None:
    """Repair keeps valid canonical optional fields when legacy extras exist."""

    path = tmp_path / "legacy.json"
    _write_json(
        path,
        {
            "actor": "worker-1",
            "content": "Legacy record with valid canonical fields.",
            "entities": ["quantum_cognition", "SC-NEUROCORE", "SC-NEUROCORE"],
            "kind": "decision",
            "project": "SC-NEUROCORE",
            "source_ref": "docs/internal/AUDIT_INDEX.md#memory",
            "timestamp": 1783613599,
            "unix_epoch": 1783613599,
        },
    )

    repaired = audit_tool.repair_stimulus_file(path, "SC-NEUROCORE")

    assert repaired["actor"] == "system"
    assert repaired["entities"] == ["SC-NEUROCORE", "quantum_cognition"]
    assert repaired["kind"] == "decision"
    assert repaired["source_ref"] == "docs/internal/AUDIT_INDEX.md#memory"


def test_repair_uses_fallback_content_and_requires_timestamp(tmp_path: Path) -> None:
    """Repair has deterministic fallback text but still requires time provenance."""

    content_path = tmp_path / "content.json"
    missing_time_path = tmp_path / "missing_time.json"
    _write_json(
        content_path, {"actor": "codex", "project": "SC-NEUROCORE", "timestamp": 1783613599}
    )
    _write_json(missing_time_path, {"actor": "codex", "project": "SC-NEUROCORE"})

    repaired = audit_tool.repair_stimulus_file(content_path, "SC-NEUROCORE")

    assert repaired["content"] == (
        "Legacy SC-NEUROCORE SNN memory record normalised to the canonical write schema."
    )
    with pytest.raises(ValueError, match="valid timestamp"):
        audit_tool.repair_stimulus_file(missing_time_path, "SC-NEUROCORE")
