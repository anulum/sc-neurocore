# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — native documentation policy repository fixtures

"""Build real isolated Git inputs and invoke the actual documentation scope CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTED_SOURCE = '"""Document the repository fixture public API."""\n'


def fixture_git(root: Path, *arguments: str) -> str:
    """Run real fixture Git commands without inheriting canonical commit hooks.

    Parameters
    ----------
    root : Path
        Disposable test repository, never the canonical product checkout.
    *arguments : str
        Actual native Git arguments.

    Returns
    -------
    str
        Successful native standard output.
    """
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Policy Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *arguments,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()


def write_policy(root: Path, paths: Sequence[str], minimum: int = 20) -> Path:
    """Write candidate TOML with an explicit file cohort and documentation floor.

    Parameters
    ----------
    root : Path
        Fixture repository root.
    paths : sequence of str
        Exact enrolled repository-relative paths.
    minimum : int, default 20
        Candidate minimum docstring length in characters.

    Returns
    -------
    Path
        Actual policy input path used by the public CLI.
    """
    path = root / "docs/docstring_policy.toml"
    path.parent.mkdir(exist_ok=True)
    text = f"[quality]\nmin_docstring_chars = {minimum}\nexpected_file_count = {len(paths)}\n"
    for name in paths:
        text += f"\n[[file]]\npath = {json.dumps(name)}\n"
    path.write_text(text, encoding="utf-8")
    return path


def repository_fixture(parent: Path) -> Path:
    """Create and commit a real original policy and Python source in a test repo.

    Parameters
    ----------
    parent : Path
        Pytest-owned temporary directory on the configured working disk.

    Returns
    -------
    Path
        Repository containing an immutable original policy commit.
    """
    root = parent / "repository"
    root.mkdir()
    (root / "original.py").write_text(DOCUMENTED_SOURCE, encoding="utf-8")
    write_policy(root, ["original.py"])
    fixture_git(root, "init", "--initial-branch=main")
    fixture_git(root, "add", "original.py", "docs/docstring_policy.toml")
    fixture_git(root, "commit", "--message", "Original documentation policy fixture")
    return root


def run_guard(
    root: Path, environment: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke the production CLI with actual Git and explicit fixture CI inputs.

    Parameters
    ----------
    root : Path
        Real fixture repository to validate.
    environment : mapping of str to str, optional
        Controlled event inputs; inherited hosted-CI inputs are cleared first.

    Returns
    -------
    subprocess.CompletedProcess
        Actual finite CLI outcome, including both diagnostic streams.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")}
    env.update(environment or {})
    return subprocess.run(
        [sys.executable, "-B", "-m", "tools.docstring_policy_guard", "--repo", str(root)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
    )
