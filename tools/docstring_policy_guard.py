# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — required documentation scope guard

"""Refuse docstring scope shrinkage and unenrolled changed Python source files."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from tools.docstring_policy_git import POLICY_PATH, baseline_commit, changed_python_paths, git_bytes
from tools.docstring_policy_scope import PolicyError, compare_policies, parse_policy


@dataclass(frozen=True)
class ScopeCheck:
    """Retain original and candidate policy hashes with actual scope diagnostics."""

    baseline_commit: str
    baseline_sha256: str
    candidate_sha256: str
    enrolled_files: int
    changed_files: int
    violations: tuple[str, ...]


def check_scope(root: Path, environment: Mapping[str, str] | None = None) -> ScopeCheck:
    """Compare working-tree rules to trusted Git history and changed source paths.

    Parameters
    ----------
    root : Path
        Exact repository root containing the candidate policy.
    environment : mapping of str to str, optional
        CI event environment; the actual process environment is used by default.

    Returns
    -------
    ScopeCheck
        Provenance and scope violations; native documentation checks are separate.

    Raises
    ------
    PolicyError
        If a required policy, Git query or CI provenance input is unavailable.
    """
    root = root.resolve()
    commit = baseline_commit(root, os.environ if environment is None else environment)
    baseline = parse_policy(git_bytes(root, "show", f"{commit}:{POLICY_PATH}"))
    try:
        candidate = parse_policy((root / POLICY_PATH).read_bytes())
    except OSError as exc:
        raise PolicyError("Candidate docstring policy could not be read.") from exc
    changed = changed_python_paths(root, commit)
    violations = compare_policies(candidate, baseline, changed)
    for relative in candidate.files:
        source = root / relative
        if not source.is_file() or not source.resolve().is_relative_to(root):
            violations.append(f"Policy source is unavailable inside the repository: {relative}")
    return ScopeCheck(
        commit,
        baseline.sha256,
        candidate.sha256,
        len(candidate.files),
        len(changed),
        tuple(violations),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scope guard with explicit refusal and unavailable-input exit codes.

    Parameters
    ----------
    argv : sequence of str, optional
        CLI arguments; process arguments are used when omitted.

    Returns
    -------
    int
        Zero for accepted scope, one for violations, two for unavailable inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = check_scope(args.repo)
    except PolicyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if result.violations:
        print("\n".join(result.violations), file=sys.stderr)
        return 1
    print(
        f"Docstring scope accepted: {result.enrolled_files} files, "
        f"{result.changed_files} changed Python files; baseline {result.baseline_commit}; "
        f"policy SHA-256 {result.candidate_sha256}; baseline policy SHA-256 {result.baseline_sha256}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
