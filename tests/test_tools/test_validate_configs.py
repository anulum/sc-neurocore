# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configuration validator tests

"""Tests for the repository configuration validator CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_configs


def _write_minimal_repo(root: Path, *, pyproject: str) -> None:
    """Create the minimal tree required by ``validate_configs``."""
    (root / "src/sc_neurocore").mkdir(parents=True)
    (root / "hdl").mkdir()
    (root / "tests").mkdir()
    (root / "docs/guides").mkdir(parents=True)
    (root / "docs/API_REFERENCE.md").write_text("# API\n", encoding="utf-8")
    (root / "docs/guides/USER_MANUAL.md").write_text("# Guide\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")


def test_validate_project_structure_accepts_required_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The structure validator accepts the required repository paths."""
    _write_minimal_repo(tmp_path, pyproject="[project]\nname='demo'\nversion='1.0'\n")

    assert validate_configs.validate_project_structure(tmp_path)
    assert "Structure Validation: PASSED" in capsys.readouterr().out


def test_validate_project_structure_reports_missing_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The structure validator fails closed when a required path is absent."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    assert not validate_configs.validate_project_structure(tmp_path)
    output = capsys.readouterr().out
    assert "[MISSING] src/sc_neurocore" in output
    assert "Structure Validation: FAILED" in output


def test_validate_pyproject_accepts_valid_toml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pyproject validator parses TOML and prints project metadata."""
    _write_minimal_repo(tmp_path, pyproject="[project]\nname='demo'\nversion='1.0'\n")

    assert validate_configs.validate_pyproject(tmp_path)
    output = capsys.readouterr().out
    assert "Name: demo" in output
    assert "Version: 1.0" in output


def test_validate_pyproject_rejects_malformed_toml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pyproject validator returns ``False`` when TOML parsing fails."""
    _write_minimal_repo(tmp_path, pyproject="[project\n")

    assert not validate_configs.validate_pyproject(tmp_path)
    assert "[ERROR] Parsing failed:" in capsys.readouterr().out


def test_validate_pyproject_rejects_malformed_project_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pyproject validator fails closed when ``project`` is not a table."""
    _write_minimal_repo(tmp_path, pyproject='project = "bad"\n')

    assert not validate_configs.validate_pyproject(tmp_path)
    assert "project must be a table" in capsys.readouterr().out


def test_validate_pyproject_skips_strict_parse_without_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The legacy no-parser path remains non-fatal for older Python setups."""
    _write_minimal_repo(tmp_path, pyproject="[project\n")
    monkeypatch.setattr(validate_configs, "_load_toml_module", lambda: None)

    assert validate_configs.validate_pyproject(tmp_path)
    assert "skipping strict parse check" in capsys.readouterr().out


def test_resolve_project_root_accepts_repository_root(tmp_path: Path) -> None:
    """Root detection returns the current directory when ``src`` is present."""
    (tmp_path / "src").mkdir()

    assert validate_configs._resolve_project_root(tmp_path) == tmp_path


def test_main_accepts_invocation_from_scripts_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI resolves the project root when invoked from ``scripts``."""
    _write_minimal_repo(tmp_path, pyproject="[project]\nname='demo'\nversion='1.0'\n")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    monkeypatch.chdir(scripts_dir)

    assert validate_configs.main() == 0


def test_main_returns_nonzero_for_invalid_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI returns a non-zero exit code when validation fails."""
    monkeypatch.chdir(tmp_path)

    assert validate_configs.main() == 1
