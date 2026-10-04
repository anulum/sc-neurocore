# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real documentation cohort and threshold refusal contracts

"""Exercise scope acceptance and invalid candidates through real Git and CLI inputs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tests.docstring_policy_guard_support import (
    DOCUMENTED_SOURCE,
    repository_fixture,
    run_guard,
    write_policy,
)
from tools.docstring_policy_guard import check_scope, main
from tools.docstring_policy_scope import PolicyError


def test_original_and_enrolled_addition_preserve_native_provenance(tmp_path: Path) -> None:
    """Accept valid additions while binding both policy versions to actual bytes."""
    root = repository_fixture(tmp_path)
    original = (root / "docs/docstring_policy.toml").read_bytes()
    assert run_guard(root).returncode == 0
    (root / "new.py").write_text(DOCUMENTED_SOURCE, encoding="utf-8")
    policy = write_policy(root, ["original.py", "new.py"], minimum=30)
    result = check_scope(root, {})
    assert result.violations == ()
    assert result.changed_files == 1 and result.enrolled_files == 2
    assert result.baseline_sha256 == hashlib.sha256(original).hexdigest()
    assert result.candidate_sha256 == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert run_guard(root).returncode == 0


@pytest.mark.parametrize("candidate", ["shrink", "minimum", "module", "allowance"])
def test_original_rules_cannot_be_weakened(tmp_path: Path, candidate: str) -> None:
    """Refuse joint count/list shrinkage, weaker floors and newly waived symbols."""
    root = repository_fixture(tmp_path)
    policy = root / "docs/docstring_policy.toml"
    if candidate == "shrink":
        (root / "replacement.py").write_text(DOCUMENTED_SOURCE, encoding="utf-8")
        write_policy(root, ["replacement.py"])
    elif candidate == "minimum":
        write_policy(root, ["original.py"], minimum=1)
    else:
        extra = (
            "require_module_docstring = false\n"
            if candidate == "module"
            else 'allow_missing = ["undocumented"]\n'
        )
        policy.write_text(policy.read_text() + extra, encoding="utf-8")
    assert check_scope(root, {}).violations
    refused = run_guard(root)
    assert refused.returncode == 1
    assert "Traceback" not in refused.stderr


@pytest.mark.parametrize(
    "directory", ["src", "tests", "tools", "examples", "research", "cosim", "bridge", "hdl"]
)
def test_new_python_files_in_every_root_require_enrollment(tmp_path: Path, directory: str) -> None:
    """Discover untracked Python inputs even in roots excluded by ordinary Ruff."""
    root = repository_fixture(tmp_path)
    folder = root / directory
    folder.mkdir()
    relative = f"{directory}/new.py"
    (root / relative).write_text("def undocumented():\n    return 1\n", encoding="utf-8")
    result = check_scope(root, {})
    assert result.changed_files == 1
    assert f"Missing required policy file: {relative}" in result.violations
    assert run_guard(root).returncode == 1


def test_changed_tracked_and_staged_python_are_enrolled(tmp_path: Path) -> None:
    """Preserve tracked changes and staged additions in the complete Git cohort."""
    from tests.docstring_policy_guard_support import fixture_git

    root = repository_fixture(tmp_path)
    (root / "original.py").write_text(DOCUMENTED_SOURCE + "VALUE = 2\n", encoding="utf-8")
    (root / "staged.py").write_text(DOCUMENTED_SOURCE, encoding="utf-8")
    fixture_git(root, "add", "staged.py")
    result = check_scope(root, {})
    assert result.changed_files == 2
    assert result.violations == ("Missing required policy file: staged.py",)
    write_policy(root, ["original.py", "staged.py"])
    assert run_guard(root).returncode == 0


@pytest.mark.parametrize(
    "content",
    [
        "broken = [",
        "file = []",
        "[quality]\nmin_docstring_chars = true\nexpected_file_count = 1",
        "[quality]\nmin_docstring_chars = 20\nexpected_file_count = false",
        "[quality]\nmin_docstring_chars = 20\nexpected_file_count = 0\nfile = []",
    ],
)
def test_malformed_and_empty_policy_refuse_without_success(tmp_path: Path, content: str) -> None:
    """Reject parser failures, absent tables and invalid scalar policy types."""
    root = repository_fixture(tmp_path)
    (root / "docs/docstring_policy.toml").write_text(content, encoding="utf-8")
    with pytest.raises(PolicyError):
        check_scope(root, {})
    assert run_guard(root).returncode == 2


@pytest.mark.parametrize(
    "entry",
    [
        'path = "original.py"\nrequire_module_docstring = "false"',
        'path = "original.py"\nallow_missing = true',
        'path = "original.py"\nallow_missing = [""]',
        'path = "original.py"\nallow_missing = [42]',
        'path = "original.py"\nallow_missing = ["method", "method"]',
        'path = "../outside.py"',
        'path = "/outside.py"',
        'path = "./original.py"',
        'path = "not_python.txt"',
        'path = ""',
        "path = 42",
        'path = "windows\\\\source.py"',
        'path = "bad\\nsource.py"',
    ],
)
def test_invalid_entry_shape_is_not_a_qualified_policy(tmp_path: Path, entry: str) -> None:
    """Refuse unsafe paths, malformed module requirements and duplicate allowances."""
    root = repository_fixture(tmp_path)
    policy = root / "docs/docstring_policy.toml"
    policy.write_text(
        "[quality]\nmin_docstring_chars = 20\nexpected_file_count = 1\n[[file]]\n" + entry + "\n"
    )
    with pytest.raises(PolicyError):
        check_scope(root, {})
    assert run_guard(root).returncode == 2


def test_missing_duplicate_and_external_source_inputs_refuse(tmp_path: Path) -> None:
    """Reject missing sources, duplicate cohorts and sources linked outside the repo."""
    root = repository_fixture(tmp_path)
    (root / "original.py").unlink()
    assert check_scope(root, {}).violations
    assert run_guard(root).returncode == 1
    outside = tmp_path / "outside.py"
    outside.write_text(DOCUMENTED_SOURCE, encoding="utf-8")
    (root / "original.py").symlink_to(outside)
    assert check_scope(root, {}).violations
    assert run_guard(root).returncode == 1
    write_policy(root, ["original.py", "original.py"])
    assert run_guard(root).returncode == 2
    (root / "docs/docstring_policy.toml").unlink()
    with pytest.raises(PolicyError, match="Candidate docstring policy could not be read"):
        check_scope(root, {})
    assert run_guard(root).returncode == 2


def test_missing_git_unborn_history_and_wrong_root_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require actual native Git, an original commit and the exact repository root."""
    from tests.docstring_policy_guard_support import fixture_git

    root = repository_fixture(tmp_path)
    assert run_guard(root, {"PATH": ""}).returncode == 2
    with monkeypatch.context() as environment:
        environment.setenv("PATH", "")
        with pytest.raises(PolicyError, match="Native Git input could not be read"):
            check_scope(root, {})
    assert run_guard(root / "docs").returncode == 2
    with pytest.raises(PolicyError, match="exact repository root"):
        check_scope(root / "docs", {})
    unborn = tmp_path / "unborn"
    unborn.mkdir()
    fixture_git(unborn, "init", "--initial-branch=main")
    assert run_guard(unborn).returncode == 2
    assert run_guard(tmp_path).returncode == 2


def test_mismatched_count_and_non_utf8_policy_refuse(tmp_path: Path) -> None:
    """Reject missing declared entries and invalid bytes through the public checker."""
    root = repository_fixture(tmp_path)
    policy = root / "docs/docstring_policy.toml"
    policy.write_text(
        policy.read_text().replace("expected_file_count = 1", "expected_file_count = 2")
    )
    with pytest.raises(PolicyError, match="count must match"):
        check_scope(root, {})
    assert run_guard(root).returncode == 2
    policy.write_bytes(b"\xff")
    with pytest.raises(PolicyError, match="UTF-8 TOML"):
        check_scope(root, {})
    assert run_guard(root).returncode == 2


def test_stronger_original_floor_and_individual_allowances_are_preserved(tmp_path: Path) -> None:
    """Preserve a stronger original floor and reject exchanging old for new debt."""
    from tests.docstring_policy_guard_support import fixture_git

    root = repository_fixture(tmp_path)
    policy = write_policy(root, ["original.py"], minimum=30)
    policy.write_text(policy.read_text() + 'allow_missing = ["old_method"]\n')
    fixture_git(root, "add", "docs/docstring_policy.toml")
    fixture_git(root, "commit", "--message", "Established stronger policy fixture")
    assert check_scope(root, {}).violations == ()
    policy.write_text(
        policy.read_text().replace("min_docstring_chars = 30", "min_docstring_chars = 25")
    )
    assert check_scope(root, {}).violations
    policy.write_text(
        policy.read_text()
        .replace("min_docstring_chars = 25", "min_docstring_chars = 30")
        .replace("old_method", "new_method")
    )
    assert check_scope(root, {}).violations == ("New missing-symbol allowance: original.py",)
    write_policy(root, ["original.py"], minimum=30)
    assert check_scope(root, {}).violations == ()
    assert run_guard(root).returncode == 0


def test_native_path_encoding_failure_and_cli_return_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject undecodable native paths and expose real CLI success/refusal results."""
    root = repository_fixture(tmp_path)
    assert main(["--repo", str(root)]) == 0
    assert "Docstring scope accepted" in capsys.readouterr().out
    write_policy(root, ["original.py"], minimum=1)
    assert main(["--repo", str(root)]) == 1
    assert "trusted floor" in capsys.readouterr().err
    write_policy(root, ["original.py"])
    bad = os.fsencode(root) + b"/invalid\xff.py"
    descriptor = os.open(bad, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    assert main(["--repo", str(root)]) == 2
    assert "UTF-8" in capsys.readouterr().err
