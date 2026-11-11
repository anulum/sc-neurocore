# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — preflight gates over the commits a push would add

"""Exercise the preflight's push-range gates on real Git repositories.

The hosted Pre-commit run checks every file and compares the documentation
policy with the commit a push starts from. The preflight finds that commit and
checks the same range before the push leaves the machine.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

import tools.preflight
from tools.preflight import (
    DEFAULT_PUSH_TARGET,
    HOOK_CONFIGURATION_PATHS,
    push_base,
    range_gates,
    run_gate,
    main,
    selected_gates,
)


def _git(root: Path, *arguments: str) -> str:
    """Run a real Git command in a fixture repository and return its output."""
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Preflight Fixture",
            "-c",
            "user.email=preflight@example.invalid",
            *arguments,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


def _commit(root: Path, name: str) -> str:
    """Commit one file and return the new head."""
    (root / name).write_text(f"{name}\n", encoding="utf-8")
    _git(root, "add", name)
    _git(root, "commit", "--quiet", "--message", f"Add {name}")
    return _git(root, "rev-parse", "HEAD")


def _clone_with_origin(tmp_path: Path) -> tuple[Path, str]:
    """Create a bare origin with one commit on main and a clone tracking it."""
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--quiet", "--initial-branch=main")
    pushed = _commit(seed, "pushed.txt")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "--quiet", "--bare", str(seed), str(origin))
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "--quiet", str(origin), str(clone))
    return clone, pushed


def test_the_base_is_the_merge_base_with_the_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local commits after the pushed head are the range; the pushed head is the base."""
    clone, pushed = _clone_with_origin(tmp_path)
    _commit(clone, "local.txt")
    monkeypatch.chdir(clone)
    assert push_base() == pushed


def test_without_an_upstream_the_default_push_target_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detached checkout, as a landing worktree is, falls back to origin/main."""
    clone, pushed = _clone_with_origin(tmp_path)
    head = _commit(clone, "local.txt")
    _git(clone, "switch", "--quiet", "--detach", head)
    monkeypatch.chdir(clone)
    assert DEFAULT_PUSH_TARGET == "origin/main"
    assert push_base() == pushed


def test_no_upstream_and_no_default_target_gives_no_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without anything to push to, the base is unknown rather than guessed."""
    lone = tmp_path / "lone"
    lone.mkdir()
    _git(lone, "init", "--quiet", "--initial-branch=main")
    _commit(lone, "only.txt")
    monkeypatch.chdir(lone)
    assert push_base() is None


def test_outside_a_repository_or_without_git_there_is_no_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git failing, or Git being absent, gives no base."""
    monkeypatch.chdir(tmp_path)
    assert push_base() is None
    monkeypatch.setenv("PATH", "")
    assert push_base() is None


def test_range_gates_check_the_changed_files_and_the_policy_since_the_base() -> None:
    """Ordinary changes run the hooks over the range and the guard from the base."""
    gates = dict(range_gates("abc123", ["src/a.py", "docs/b.md"]))
    assert gates["docstring-scope-range"] == [
        "python",
        "-m",
        "tools.docstring_policy_guard",
        "--since",
        "abc123",
    ]
    assert gates["pre-commit-range"] == [
        "python",
        "-m",
        "pre_commit",
        "run",
        "--show-diff-on-failure",
        "--from-ref",
        "abc123",
        "--to-ref",
        "HEAD",
    ]


@pytest.mark.parametrize("configuration", sorted(HOOK_CONFIGURATION_PATHS))
def test_a_hook_configuration_change_checks_all_files(configuration: str) -> None:
    """A changed hook version or rule can affect every file, so all files are checked."""
    gates = dict(range_gates("abc123", ["src/a.py", configuration]))
    assert gates["pre-commit-range"][-1] == "--all-files"
    assert "--from-ref" not in gates["pre-commit-range"]


def test_selected_gates_end_with_the_range_of_this_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The range gates follow the configured gates and use this checkout's real base."""
    clone, pushed = _clone_with_origin(tmp_path)
    _commit(clone, ".typos.toml")
    monkeypatch.chdir(clone)
    gates = selected_gates(no_tests=True)
    names = [name for name, _ in gates]
    assert "pytest" not in names
    assert names[-2:] == ["docstring-scope-range", "pre-commit-range"]
    assert gates[-2][1] == ["python", "-m", "tools.docstring_policy_guard", "--since", pushed]
    assert gates[-1][1] is not None and gates[-1][1][-1] == "--all-files"
    assert "pytest" in [name for name, _ in selected_gates(no_tests=False)]


@pytest.mark.parametrize("change", ["modified", "staged", "untracked"])
def test_a_checkout_that_differs_from_head_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    change: str,
) -> None:
    """The range hooks read the working tree, but a push sends HEAD, so the tree must be clean."""
    clone, _ = _clone_with_origin(tmp_path)
    _commit(clone, "local.txt")
    if change == "untracked":
        (clone / "stray.txt").write_text("stray\n", encoding="utf-8")
    else:
        (clone / "local.txt").write_text("edited\n", encoding="utf-8")
        if change == "staged":
            _git(clone, "add", "local.txt")
    monkeypatch.chdir(clone)
    gates = selected_gates(no_tests=True)
    assert gates[-1] == ("push-range", None)
    assert "docstring-scope-range" not in [name for name, _ in gates]
    output = capsys.readouterr().out
    assert "changes a push would not send" in output
    assert ("stray.txt" if change == "untracked" else "local.txt") in output


def test_an_unreadable_checkout_state_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When Git cannot read the index, cleanliness is unknown and the range gates refuse."""
    clone, _ = _clone_with_origin(tmp_path)
    _commit(clone, "local.txt")
    index = clone / ".git" / "index"
    index.chmod(0)
    try:
        monkeypatch.chdir(clone)
        gates = selected_gates(no_tests=True)
    finally:
        index.chmod(0o644)
    assert gates[-1] == ("push-range", None)
    assert "checkout state could not be read" in capsys.readouterr().out


def test_an_unknown_base_makes_the_preflight_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a base the range cannot be checked, and the preflight refuses."""
    monkeypatch.chdir(tmp_path)
    gates = selected_gates(no_tests=True)
    assert gates[-1] == ("push-range", None)
    assert "push base could not be determined" in capsys.readouterr().out
    assert run_gate(*gates[-1]) is False


def test_the_preflight_refuses_a_push_whose_range_it_cannot_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole command runs its gates, then fails on the unknown range and names it."""
    lone = tmp_path / "lone"
    lone.mkdir()
    _git(lone, "init", "--quiet", "--initial-branch=main")
    _commit(lone, "only.txt")
    monkeypatch.chdir(lone)
    monkeypatch.setattr(tools.preflight, "GATES", [("interpreter", [sys.executable, "-c", "pass"])])
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--no-tests"])
    assert main() == 1
    output = capsys.readouterr().out
    assert "PASS: interpreter" in output
    assert "PREFLIGHT: 1 passed, 1 failed" in output
    assert "FAILED: push-range" in output
