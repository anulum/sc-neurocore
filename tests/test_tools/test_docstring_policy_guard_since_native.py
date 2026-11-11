# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — explicit local range baseline for the documentation scope guard

"""Exercise the guard's explicit ``--since`` baseline on real Git history.

A push is checked by the hosted workflow against the commit the push starts
from, so every Python file changed in the pushed range must be enrolled. The
explicit baseline lets the local pre-push gate apply the same range.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.docstring_policy_guard_support import (
    REPO_ROOT,
    fixture_git,
    repository_fixture,
    write_policy,
)
from tools.docstring_policy_guard import check_scope
from tools.docstring_policy_scope import PolicyError


def _guard_since(
    root: Path, since: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the production CLI with an explicit baseline and no inherited CI inputs."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")}
    env.update(environment or {})
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.docstring_policy_guard",
            "--repo",
            str(root),
            "--since",
            since,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
    )


def _commit_source(root: Path, name: str) -> str:
    """Commit one documented Python file and return the new head."""
    (root / name).write_text('"""Document a committed source file in the range."""\n')
    fixture_git(root, "add", name)
    fixture_git(root, "commit", "--message", f"Add {name}")
    return fixture_git(root, "rev-parse", "HEAD")


def test_range_since_the_push_base_requires_every_committed_source(tmp_path: Path) -> None:
    """A file committed earlier in the range is required, as the hosted push check requires it."""
    root = repository_fixture(tmp_path)
    base = fixture_git(root, "rev-parse", "HEAD")
    _commit_source(root, "first.py")
    _commit_source(root, "second.py")
    # The default local mode compares with HEAD only and sees nothing changed.
    assert check_scope(root, {}).violations == ()
    result = check_scope(root, {}, since=base)
    assert result.baseline_commit == base
    assert result.violations == (
        "Missing required policy file: first.py",
        "Missing required policy file: second.py",
    )
    refused = _guard_since(root, base)
    assert refused.returncode == 1
    assert "Missing required policy file: first.py" in refused.stderr
    write_policy(root, ["original.py", "first.py", "second.py"])
    accepted = _guard_since(root, base)
    assert accepted.returncode == 0
    assert f"baseline {base}" in accepted.stdout
    assert "2 changed Python files" in accepted.stdout


def test_since_head_matches_the_default_local_mode(tmp_path: Path) -> None:
    """With nothing to push, the explicit baseline is the checkout itself."""
    root = repository_fixture(tmp_path)
    head = fixture_git(root, "rev-parse", "HEAD")
    assert check_scope(root, {}, since="HEAD") == check_scope(root, {})
    assert _guard_since(root, head).returncode == 0


@pytest.mark.parametrize("revision", ["not-a-revision", "-p", ""])
def test_unknown_or_option_like_baselines_are_unavailable_inputs(
    tmp_path: Path, revision: str
) -> None:
    """A baseline that names no commit is refused as unavailable, never treated as HEAD."""
    root = repository_fixture(tmp_path)
    with pytest.raises(PolicyError):
        check_scope(root, {}, since=revision)
    assert _guard_since(root, revision).returncode == 2


def test_a_commit_the_checkout_does_not_descend_from_is_refused(tmp_path: Path) -> None:
    """A baseline from another branch cannot define the range of this checkout."""
    root = repository_fixture(tmp_path)
    fixture_git(root, "switch", "--create", "other")
    other = _commit_source(root, "other.py")
    fixture_git(root, "switch", "main")
    _commit_source(root, "mainline.py")
    with pytest.raises(PolicyError):
        check_scope(root, {}, since=other)
    assert _guard_since(root, other).returncode == 2


def test_hosted_ci_refuses_an_explicit_baseline(tmp_path: Path) -> None:
    """CI keeps its event baseline; an explicit local baseline is refused there."""
    root = repository_fixture(tmp_path)
    head = fixture_git(root, "rev-parse", "HEAD")
    with pytest.raises(PolicyError, match="CI uses its event baseline"):
        check_scope(root, {"GITHUB_ACTIONS": "true"}, since=head)
    refused = _guard_since(root, head, {"GITHUB_ACTIONS": "true"})
    assert refused.returncode == 2
    assert "CI uses its event baseline" in refused.stderr


def test_the_explicit_baseline_requires_the_exact_repository_root(tmp_path: Path) -> None:
    """A subdirectory cannot stand in for the repository root."""
    root = repository_fixture(tmp_path)
    head = fixture_git(root, "rev-parse", "HEAD")
    with pytest.raises(PolicyError, match="exact repository root"):
        check_scope(root / "docs", {}, since=head)
