# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — immutable CI documentation baseline contracts

"""Exercise real Git history against push, pull-request and manual CI event inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.docstring_policy_guard_support import (
    fixture_git,
    repository_fixture,
    run_guard,
    write_policy,
)
from tools.docstring_policy_guard import check_scope
from tools.docstring_policy_scope import PolicyError


def _event(root: Path, name: str, payload: object) -> dict[str, str]:
    path = root.parent / "event.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_EVENT_NAME": name,
        "GITHUB_EVENT_PATH": str(path),
        "GITHUB_SHA": fixture_git(root, "rev-parse", "HEAD"),
    }


@pytest.mark.parametrize("name", ["push", "pull_request", "workflow_dispatch"])
def test_ci_uses_original_policy_before_a_committed_weakening(tmp_path: Path, name: str) -> None:
    """A committed weaker floor must fail against the event's original Git policy."""
    root = repository_fixture(tmp_path)
    original = fixture_git(root, "rev-parse", "HEAD")
    write_policy(root, ["original.py"], minimum=1)
    fixture_git(root, "add", "docs/docstring_policy.toml")
    fixture_git(root, "commit", "--message", "Candidate policy fixture")
    head = fixture_git(root, "rev-parse", "HEAD")
    payload: object = (
        {"before": original, "after": head}
        if name == "push"
        else {"pull_request": {"base": {"sha": original}}}
    )
    environment = _event(root, name, payload)
    result = check_scope(root, environment)
    assert result.baseline_commit == original
    assert result.violations
    assert run_guard(root, environment).returncode == 1


def test_push_accepts_enrolled_committed_source_and_rejects_omission(tmp_path: Path) -> None:
    """The event diff includes new tracked code even when the working tree is clean."""
    root = repository_fixture(tmp_path)
    original = fixture_git(root, "rev-parse", "HEAD")
    (root / "committed.py").write_text('"""Document a genuinely committed new source file."""\n')
    fixture_git(root, "add", "committed.py")
    fixture_git(root, "commit", "--message", "New source fixture")
    head = fixture_git(root, "rev-parse", "HEAD")
    environment = _event(root, "push", {"before": original, "after": head})
    assert check_scope(root, environment).violations == (
        "Missing required policy file: committed.py",
    )
    assert run_guard(root, environment).returncode == 1
    write_policy(root, ["original.py", "committed.py"])
    assert run_guard(root, environment).returncode == 0


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"before": "0" * 40},
        {"before": "HEAD"},
        {"before": "f" * 40, "after": "e" * 40},
    ],
)
def test_missing_or_unavailable_ci_original_never_falls_back_to_head(
    tmp_path: Path, payload: object
) -> None:
    """Malformed, mutable, unborn or unavailable baselines cannot become a self-check."""
    root = repository_fixture(tmp_path)
    environment = _event(root, "push", payload)
    with pytest.raises(PolicyError):
        check_scope(root, environment)
    refused = run_guard(root, environment)
    assert refused.returncode == 2
    assert "Traceback" not in refused.stderr


@pytest.mark.parametrize(
    "failure",
    [
        "missing_event",
        "unreadable_event",
        "invalid_json",
        "invalid_utf8",
        "wrong_checkout",
        "unknown_event",
        "self_baseline",
        "missing_pr_base",
        "unavailable_base",
        "no_dispatch_parent",
    ],
)
def test_ci_provenance_errors_refuse_actual_cli(tmp_path: Path, failure: str) -> None:
    """Require readable event identity, current checkout and an available older commit."""
    root = repository_fixture(tmp_path)
    head = fixture_git(root, "rev-parse", "HEAD")
    environment = _event(root, "push", {"before": head, "after": head})
    event = Path(environment["GITHUB_EVENT_PATH"])
    if failure == "missing_event":
        environment.pop("GITHUB_EVENT_PATH")
    elif failure == "unreadable_event":
        event.unlink()
    elif failure == "invalid_json":
        event.write_text("{")
    elif failure == "invalid_utf8":
        event.write_bytes(b"\xff")
    elif failure == "wrong_checkout":
        environment["GITHUB_SHA"] = "e" * 40
    elif failure == "unknown_event":
        environment["GITHUB_EVENT_NAME"] = "unqualified_event"
    elif failure == "missing_pr_base":
        environment["GITHUB_EVENT_NAME"] = "pull_request"
    elif failure == "unavailable_base":
        event.write_text(json.dumps({"before": "f" * 40, "after": head}))
    elif failure == "no_dispatch_parent":
        environment["GITHUB_EVENT_NAME"] = "workflow_dispatch"
    with pytest.raises(PolicyError):
        check_scope(root, environment)
    assert run_guard(root, environment).returncode == 2


def test_available_unrelated_git_commit_is_not_a_ci_baseline(tmp_path: Path) -> None:
    """Require ancestry even when the supplied original commit really exists."""
    root = repository_fixture(tmp_path)
    head = fixture_git(root, "rev-parse", "HEAD")
    unrelated = fixture_git(root, "commit-tree", "HEAD^{tree}", "-m", "Unrelated history fixture")
    environment = _event(root, "push", {"before": unrelated, "after": head})
    with pytest.raises(PolicyError, match="Native Git input could not be read"):
        check_scope(root, environment)
    assert run_guard(root, environment).returncode == 2


def test_pull_request_merge_commit_uses_actual_base_policy(tmp_path: Path) -> None:
    """Read the original base under an actual two-parent pull-request merge commit."""
    root = repository_fixture(tmp_path)
    base = fixture_git(root, "rev-parse", "HEAD")
    write_policy(root, ["original.py"], minimum=1)
    fixture_git(root, "add", "docs/docstring_policy.toml")
    fixture_git(root, "commit", "--message", "Pull request candidate fixture")
    candidate = fixture_git(root, "rev-parse", "HEAD")
    merge = fixture_git(
        root, "commit-tree", "HEAD^{tree}", "-p", base, "-p", candidate, "-m", "Merge fixture"
    )
    fixture_git(root, "update-ref", "HEAD", merge)
    environment = _event(root, "pull_request", {"pull_request": {"base": {"sha": base}}})
    result = check_scope(root, environment)
    assert result.baseline_commit == base and result.violations
    assert run_guard(root, environment).returncode == 1
