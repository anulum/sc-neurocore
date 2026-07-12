# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NIR document CLI tests

"""Exercise SC-NIR document operations through the public CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.cli_test_support import run_cli


def test_scnir_without_action_prints_complete_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing action fails closed and advertises every supported workflow."""
    assert run_cli("scnir") == 1
    output = capsys.readouterr().out
    for action in ("validate", "upgrade", "export", "audit-hdl", "compatibility", "closure-audit"):
        assert action in output


def test_scnir_compatibility_defaults_to_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compatibility auditing accepts the current directory as its evidence root."""
    import sc_neurocore.ir as ir

    observed: list[Path] = []
    monkeypatch.setattr(
        ir,
        "validate_scnir_compatibility_matrix",
        lambda *, evidence_root: observed.append(evidence_root),
    )

    assert run_cli("scnir", "compatibility") == 0
    assert observed == [Path.cwd()]
    assert "SC-NIR compatibility matrix valid" in capsys.readouterr().out


def test_scnir_closure_audit_defaults_to_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Closure auditing reports a path-free summary without requiring an output file."""
    import sc_neurocore.ir as ir

    observed: list[Path] = []

    def build_report(*, evidence_root: Path) -> dict[str, int]:
        observed.append(evidence_root)
        return {"primitive_count": 3, "audit_evidence_file_count": 7}

    monkeypatch.setattr(ir, "build_scnir_compatibility_audit", build_report)

    assert run_cli("scnir", "closure-audit") == 0
    assert observed == [Path.cwd()]
    assert "3 primitive(s), 7 evidence file(s)" in capsys.readouterr().out


def test_scnir_closure_audit_reports_invalid_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Closure-audit validation failures return status one."""
    import sc_neurocore.ir as ir

    def fail_audit(*, evidence_root: Path) -> object:
        del evidence_root
        raise ValueError("invalid closure evidence")

    monkeypatch.setattr(ir, "build_scnir_compatibility_audit", fail_audit)

    assert run_cli("scnir", "closure-audit") == 1
    assert "invalid closure evidence" in capsys.readouterr().out


def test_scnir_upgrade_requires_explicit_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Upgrade never overwrites its source document implicitly."""
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")

    assert run_cli("scnir", "upgrade", str(source)) == 1
    assert "requires --output" in capsys.readouterr().out


def test_scnir_upgrade_rejects_non_object_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Upgrade rejects JSON arrays before schema migration."""
    source = tmp_path / "source.json"
    source.write_text("[]", encoding="utf-8")

    assert run_cli("scnir", "upgrade", str(source), "--output", str(tmp_path / "out.json")) == 1
    assert "must be a JSON object" in capsys.readouterr().out


def test_scnir_audit_hdl_can_report_without_writing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A valid HDL handoff can be audited as a console-only operation."""
    import sc_neurocore.ir as ir

    report = SimpleNamespace(stream_count=2, source_module_count=3)
    monkeypatch.setattr(ir, "audit_scnir_hdl_handoff", lambda _path: report)

    assert run_cli("scnir", "audit-hdl", str(tmp_path)) == 0
    assert "2 stream(s), 3 source module(s)" in capsys.readouterr().out


def test_scnir_export_requires_explicit_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Export never invents or overwrites a destination path."""
    assert run_cli("scnir", "export", str(tmp_path / "model.nir")) == 1
    assert "requires --output" in capsys.readouterr().out


def test_scnir_export_reports_conversion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Conversion errors cross the command boundary as a non-zero status."""
    import sc_neurocore.ir as ir

    def fail_export(*_args: object, **_kwargs: object) -> object:
        raise ValueError("invalid conversion fixture")

    monkeypatch.setattr(ir, "export_scnir_from_nir", fail_export)

    assert (
        run_cli(
            "scnir",
            "export",
            str(tmp_path / "model.nir"),
            "--output",
            str(tmp_path / "model.scnir.json"),
        )
        == 1
    )
    assert "invalid conversion fixture" in capsys.readouterr().out
