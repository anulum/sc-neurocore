# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — docstring scope guard: rows of deleted files

"""Let a staged deletion of an enrolled file take its policy row, and nothing else."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.docstring_policy_guard_support import (
    DOCUMENTED_SOURCE,
    fixture_git,
    repository_fixture,
    run_guard,
    write_policy,
)
from tools.docstring_policy_guard import check_scope, main
from tools.docstring_policy_scope import PolicyError


def _two_file_fixture(parent: Path) -> Path:
    """Commit a baseline that enrols ``original.py`` and ``kept.py``."""
    root = repository_fixture(parent)
    (root / "kept.py").write_text(DOCUMENTED_SOURCE, encoding="utf-8")
    write_policy(root, ["original.py", "kept.py"])
    fixture_git(root, "add", "kept.py", "docs/docstring_policy.toml")
    fixture_git(root, "commit", "--message", "Enrol a second fixture file")
    return root


def test_a_staged_deletion_takes_its_row_with_it(tmp_path: Path) -> None:
    """Deleting an enrolled file in the index lets its row and its count leave the policy."""
    root = _two_file_fixture(tmp_path)
    fixture_git(root, "rm", "--quiet", "original.py")
    write_policy(root, ["kept.py"])
    result = check_scope(root, {})
    assert result.violations == ()
    assert result.enrolled_files == 1
    assert run_guard(root).returncode == 0


def test_a_row_cannot_leave_while_its_file_stays(tmp_path: Path) -> None:
    """Removing the row of a file that is still tracked is refused."""
    root = _two_file_fixture(tmp_path)
    write_policy(root, ["kept.py"])
    assert check_scope(root, {}).violations == ("Missing required policy file: original.py",)
    assert run_guard(root).returncode == 1


def test_a_working_tree_deletion_does_not_count(tmp_path: Path) -> None:
    """A file removed only from the working tree is still in the index, so its row stays."""
    root = _two_file_fixture(tmp_path)
    (root / "original.py").unlink()
    write_policy(root, ["kept.py"])
    assert "Missing required policy file: original.py" in check_scope(root, {}).violations
    assert run_guard(root).returncode == 1


def test_a_deleted_file_cannot_keep_its_row(tmp_path: Path) -> None:
    """An enrolled file deleted in the index while its row stays is refused."""
    root = _two_file_fixture(tmp_path)
    fixture_git(root, "rm", "--quiet", "original.py")
    assert check_scope(root, {}).violations == (
        "Policy source is unavailable inside the repository: original.py",
    )
    assert run_guard(root).returncode == 1


def test_the_count_cannot_fall_further_than_the_deleted_rows(tmp_path: Path) -> None:
    """A count lowered past the rows that remain is not a qualified policy."""
    root = _two_file_fixture(tmp_path)
    fixture_git(root, "rm", "--quiet", "original.py")
    policy = write_policy(root, ["kept.py"])
    policy.write_text(
        policy.read_text(encoding="utf-8").replace(
            "expected_file_count = 1", "expected_file_count = 0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(PolicyError):
        check_scope(root, {})
    assert run_guard(root).returncode == 2


@pytest.mark.parametrize("enrol_new_path", [False, True])
def test_a_rename_is_a_deletion_and_an_addition(tmp_path: Path, enrol_new_path: bool) -> None:
    """A moved enrolled file may drop its old row only when its new path is enrolled."""
    root = _two_file_fixture(tmp_path)
    fixture_git(root, "mv", "original.py", "renamed.py")
    write_policy(root, ["kept.py", "renamed.py"] if enrol_new_path else ["kept.py"])
    violations = check_scope(root, {}).violations
    if enrol_new_path:
        assert violations == ()
        assert run_guard(root).returncode == 0
    else:
        assert violations == ("Missing required policy file: renamed.py",)
        assert run_guard(root).returncode == 1


def test_a_deleted_path_that_is_not_utf8_is_unavailable_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged deletion whose path is not valid UTF-8 refuses as unavailable input."""
    for name in ("GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_EVENT_PATH", "GITHUB_EVENT_NAME"):
        monkeypatch.delenv(name, raising=False)
    root = repository_fixture(tmp_path)
    bad = os.fsencode(root) + b"/invalid\xff.py"
    descriptor = os.open(bad, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    fixture_git(root, "add", "--all")
    fixture_git(root, "commit", "--message", "Track a path that is not UTF-8")
    os.remove(bad)
    fixture_git(root, "add", "--all")
    with pytest.raises(PolicyError, match="Deleted source paths must be valid UTF-8"):
        check_scope(root, {})
    assert main(["--repo", str(root)]) == 2
    assert "Deleted source paths must be valid UTF-8" in capsys.readouterr().err
