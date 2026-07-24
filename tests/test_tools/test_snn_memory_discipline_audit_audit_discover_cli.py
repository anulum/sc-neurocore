# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (audit_discover_cli) from former test_snn_memory_discipline_audit.py

from __future__ import annotations

from snn_memory_discipline_audit_support import *  # noqa: F403

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
