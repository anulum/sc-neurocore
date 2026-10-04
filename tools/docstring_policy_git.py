# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — native Git docstring baseline and changed-source discovery

"""Resolve immutable docstring baselines and complete changed Python cohorts."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import subprocess
from typing import cast

from tools.docstring_policy_scope import PolicyError

POLICY_PATH = "docs/docstring_policy.toml"


def git_bytes(root: Path, *arguments: str) -> bytes:
    """Run a bounded native Git read and reject every unsuccessful completion.

    Parameters
    ----------
    root : Path
        Repository working directory.
    *arguments : str
        Native Git command and its literal arguments.

    Returns
    -------
    bytes
        Successful native standard output, without decoding file contents.

    Raises
    ------
    PolicyError
        If Git cannot run, times out or reports failure.
    """
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        env.pop(name, None)
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", *arguments],
            cwd=root,
            env=env,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyError("Native Git input could not be read.") from exc
    if result.returncode != 0:
        raise PolicyError("Native Git input could not be read.")
    return result.stdout


def _commit(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise PolicyError("CI baseline must identify an immutable Git commit.")
    if set(value) == {"0"}:
        raise PolicyError("A new ref needs an established documentation baseline.")
    return value


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PolicyError("CI event must contain the required object fields.")
    return cast(dict[str, object], value)


def baseline_commit(root: Path, environment: Mapping[str, str]) -> str:
    """Choose local HEAD or the original commit supplied by the actual CI event.

    Parameters
    ----------
    root : Path
        Exact repository root; subdirectories cannot replace the source scope.
    environment : mapping of str to str
        Process environment, including GitHub's event identity when in CI.

    Returns
    -------
    str
        Verified available baseline commit, ancestral to the current checkout.

    Raises
    ------
    PolicyError
        If event provenance, repository identity or Git history is unavailable.
    """
    top = git_bytes(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(top).resolve() != root.resolve():
        raise PolicyError("The policy guard requires the exact repository root.")
    head = _commit(git_bytes(root, "rev-parse", "HEAD^{commit}").decode("ascii").strip())
    if environment.get("GITHUB_ACTIONS") != "true":
        return head
    if _commit(environment.get("GITHUB_SHA")) != head:
        raise PolicyError("CI checkout does not match the triggering commit.")
    event_path = environment.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise PolicyError("CI event provenance is required.")
    try:
        event = _object(json.loads(Path(event_path).read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyError("CI event provenance could not be read.") from exc
    name = environment.get("GITHUB_EVENT_NAME")
    if name == "push":
        baseline = _commit(event.get("before"))
        if _commit(event.get("after")) != head:
            raise PolicyError("Push event does not match the current checkout.")
    elif name == "pull_request":
        baseline = _commit(_object(_object(event.get("pull_request")).get("base")).get("sha"))
    elif name == "workflow_dispatch":
        baseline = _commit(git_bytes(root, "rev-parse", "HEAD^1^{commit}").decode("ascii").strip())
    else:
        raise PolicyError("This CI event has no qualified documentation baseline protocol.")
    if baseline == head:
        raise PolicyError("CI candidates cannot use their own policy as the baseline.")
    git_bytes(root, "rev-parse", "--verify", baseline + "^{commit}")
    git_bytes(root, "merge-base", "--is-ancestor", baseline, head)
    return baseline


def changed_python_paths(root: Path, baseline: str) -> set[str]:
    """Discover changed tracked and untracked Python files across all source roots.

    Parameters
    ----------
    root : Path
        Validated repository root.
    baseline : str
        Available original Git commit used by the policy comparison.

    Returns
    -------
    set of str
        Repository-relative Python paths, without test or tooling exemptions.

    Raises
    ------
    PolicyError
        If either complete native source query fails or has invalid encoding.
    """
    changed = git_bytes(
        root, "diff", "--name-only", "-z", "--diff-filter=ACMRT", baseline, "--", "*.py"
    )
    untracked = git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z", "--", "*.py")
    try:
        return {part.decode("utf-8") for part in (changed + untracked).split(b"\0") if part}
    except UnicodeError as exc:
        raise PolicyError("Changed source paths must be valid UTF-8.") from exc
