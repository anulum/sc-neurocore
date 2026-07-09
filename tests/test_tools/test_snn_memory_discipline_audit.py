# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN memory discipline audit tests

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tools import snn_memory_discipline_audit as audit_tool


def _canonical_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "actor": "codex",
        "content": "SC-NEUROCORE canonical memory discipline fixture.",
        "entities": ["SC-NEUROCORE"],
        "kind": "event",
        "project": "SC-NEUROCORE",
        "source_ref": "tests/test_tools/test_snn_memory_discipline_audit.py",
        "timestamp": 1783617021,
    }
    payload.update(overrides)
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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


def test_audit_memory_discipline_reports_directory_violations(tmp_path: Path) -> None:
    """The aggregate audit includes checked count and violation details."""

    _write_json(tmp_path / "good.json", _canonical_payload())
    _write_json(tmp_path / "bad.json", _canonical_payload(project="OTHER"))

    result = audit_tool.MemoryDisciplineAudit(
        schema_version=audit_tool.SCHEMA_VERSION,
        project="SC-NEUROCORE",
        producer_candidates=(audit_tool.ProducerCandidate("src/x.py", "emit", ("ref",)),),
        stimulus_dir=str(tmp_path),
        checked_records=2,
        violations=tuple(
            violation
            for path in sorted(tmp_path.glob("*.json"))
            for violation in audit_tool.validate_stimulus_file(path, tmp_path, "SC-NEUROCORE")
        ),
    )
    payload = result.to_json()

    assert not result.passed
    assert payload["checked_records"] == 2
    assert payload["violation_count"] == 1
    assert payload["violations"] == [
        {"path": "bad.json", "code": "invalid_project", "detail": "project must be SC-NEUROCORE"}
    ]


def test_audit_memory_discipline_builds_real_report(tmp_path: Path) -> None:
    """The aggregate builder discovers producers and validates real files."""

    _write_json(tmp_path / "good.json", _canonical_payload())

    result = audit_tool.audit_memory_discipline(Path.cwd(), tmp_path, "SC-NEUROCORE")

    assert result.checked_records == 1
    assert result.violations == ()
    assert result.passed


def test_discover_snn_producers_finds_quantum_cognition_writer() -> None:
    """Producer discovery finds the real quantum-cognition stimulus writer."""

    candidates = audit_tool.discover_snn_producers(Path.cwd())

    assert (
        audit_tool.ProducerCandidate(
            path="src/sc_neurocore/quantum_cognition/__main__.py",
            function="_emit_snn_stimulus",
            source_refs=("sc_neurocore.quantum_cognition.__main__:_emit_snn_stimulus",),
        )
        in candidates
    )


def test_discover_snn_producers_ignores_syntax_errors(tmp_path: Path) -> None:
    """Producer discovery skips tracked Python files that cannot be parsed."""

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    subprocess.run(["git", "add", "bad.py"], cwd=tmp_path, check=True, capture_output=True)

    assert audit_tool.discover_snn_producers(tmp_path) == ()


def test_cli_outputs_json_report_and_returns_failure_for_bad_record(tmp_path: Path) -> None:
    """The CLI writes machine-readable evidence and fails on violations."""

    stimulus_dir = tmp_path / "stimuli"
    stimulus_dir.mkdir()
    _write_json(stimulus_dir / "bad.json", _canonical_payload(actor="worker-1"))
    output = tmp_path / "audit.json"

    exit_code = audit_tool.main(
        [
            "--repo",
            str(Path.cwd()),
            "--stimulus-dir",
            str(stimulus_dir),
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["passed"] is False
    assert payload["producer_candidate_count"] >= 1
    assert payload["violations"] == [
        {"path": "bad.json", "code": "invalid_actor", "detail": _actor_detail()}
    ]


def test_cli_repair_outputs_passing_report(tmp_path: Path) -> None:
    """The CLI repair mode normalises legacy records before reporting."""

    stimulus_dir = tmp_path / "stimuli"
    stimulus_dir.mkdir()
    _write_json(
        stimulus_dir / "legacy.json",
        {
            "actor": "codex-seat-14753",
            "project": "SC-NEUROCORE",
            "summary": "Closed a SC-NEUROCORE audit item.",
            "timestamp": "2026-07-09T161319Z",
            "unix_epoch": 1783613599,
        },
    )

    exit_code = audit_tool.main(
        [
            "--repo",
            str(Path.cwd()),
            "--stimulus-dir",
            str(stimulus_dir),
            "--repair",
        ]
    )

    assert exit_code == 0
    assert (
        audit_tool.validate_stimulus_file(
            stimulus_dir / "legacy.json", stimulus_dir, "SC-NEUROCORE"
        )
        == ()
    )


def test_display_path_keeps_absolute_path_outside_root(tmp_path: Path) -> None:
    """Validation reports absolute paths for files outside the selected root."""

    root = tmp_path / "root"
    root.mkdir()
    path = tmp_path / "record.json"
    _write_json(path, _canonical_payload())

    assert audit_tool.validate_stimulus_file(path, root, "SC-NEUROCORE") == ()


def test_module_entrypoint_exits_with_cli_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module `__main__` path delegates to the CLI."""

    stimulus_dir = tmp_path / "stimuli"
    stimulus_dir.mkdir()
    _write_json(stimulus_dir / "record.json", _canonical_payload())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snn_memory_discipline_audit",
            "--repo",
            str(Path.cwd()),
            "--stimulus-dir",
            str(stimulus_dir),
        ],
    )
    monkeypatch.delitem(sys.modules, "tools.snn_memory_discipline_audit", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("tools.snn_memory_discipline_audit", run_name="__main__")

    assert exc_info.value.code == 0


def _actor_detail() -> str:
    return f"actor must be one of {sorted(audit_tool.CONTROLLED_ACTORS)}"
