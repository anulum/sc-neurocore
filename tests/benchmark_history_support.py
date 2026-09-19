# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — historical aggregate benchmark test support

"""Reconstruct ordered source bundles at a measured artifact's Git revision."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from tools.benchmark_evidence_gate import (
    GateFailure,
    _artefact_commit,
    _git_bytes,
    _safe_repo_path,
)


def committed_bundle_digest(
    artifact: Path,
    *,
    repo_root: Path,
    select_paths: Callable[[tuple[str, ...]], list[str]],
) -> tuple[str, int]:
    """Hash historical path names and bytes in the benchmark's exact order.

    The selector receives only files from the artifact-writing Git tree. This
    keeps later additions or removals out of a historical source bundle.
    """
    relative = artifact.resolve().relative_to(repo_root.resolve()).as_posix()
    failures: list[GateFailure] = []
    revision = _artefact_commit(repo_root, relative, "bundle", failures)
    if revision is None:
        raise AssertionError(f"unverifiable committed artifact: {failures}")
    tree_bytes = _git_bytes(repo_root, "ls-tree", "-r", "-z", "--name-only", revision)
    if tree_bytes is None:
        raise AssertionError("historical source tree is unavailable")
    tree = tuple(path.decode("utf-8") for path in tree_bytes.split(b"\0") if path)
    paths = select_paths(tree)
    if not paths or len(paths) != len(set(paths)):
        raise AssertionError("historical source selection is empty or duplicated")
    available = set(tree)
    digest = hashlib.sha256()
    for path in paths:
        if not _safe_repo_path(path) or path not in available:
            raise AssertionError(f"source missing from historical tree: {path}")
        content = _git_bytes(repo_root, "show", f"{revision}:{path}")
        if content is None:
            raise AssertionError(f"historical source bytes unavailable: {path}")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest(), len(paths)
