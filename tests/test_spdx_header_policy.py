# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SPDX header policy tests

"""Policy tests for tracked Python and Rust source headers and for JSON ownership sidecars."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools import spdx_header_audit


SOURCE_EXTENSIONS = (".py", ".rs")
"""Tracked source extensions covered by the S001/S002 audit items."""

HEADER_WINDOW_LINES = 10
"""Number of leading lines allowed for shebangs and policy header comments."""

FIXTURE_JSON_PATHSPEC = "tests/fixtures/*.json"
"""Git pathspec of the tracked fixture JSON files; its ``*`` also matches nested directories."""

SIDECAR_LICENCE = "SPDX-License-Identifier: AGPL-3.0-or-later"
"""The licence declaration every fixture JSON sidecar carries."""

SIDECAR_COPYRIGHT_RECORDS = (
    "SPDX-FileCopyrightText: 1996-2026 Miroslav Šotek <protoscience@anulum.li> (concepts)",
    "SPDX-FileCopyrightText: 2020-2026 Miroslav Šotek <protoscience@anulum.li> (code)",
)
"""The two copyright records the JSON ownership contract requires in each sidecar."""

LEGACY_JSON_LIST = "tools/json_ownership_legacy_paths.txt"
"""The tracked JSON files that predate the 2026-10-01 ownership contract and have no sidecar yet."""

VENDORED_PREFIXES = ("bridge/vendor/",)
"""Third-party trees whose JSON keeps its upstream licence through ``REUSE.toml``."""

CONTRACT_DATE = "2026-10-01T00:00:00Z"
"""The date of the JSON ownership contract; a JSON file added on or after it needs a sidecar."""

LEGACY_CEILING = 616
"""The legacy list's length when it was written; lower it as entries leave, never raise it."""


def _repo_root() -> Path:
    """Return the repository root used by the local git checkout."""
    return Path(__file__).resolve().parents[1]


def _tracked_source_files(root: Path) -> list[Path]:
    """Return tracked Python and Rust source files for SPDX verification."""
    completed = subprocess.run(
        ["git", "ls-files", *[f"*{extension}" for extension in SOURCE_EXTENSIONS]],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / line for line in completed.stdout.splitlines() if line]


def _has_spdx_header(path: Path) -> bool:
    """Return true when an SPDX marker is present near the file top."""
    leading_lines = path.read_text(encoding="utf-8").splitlines()[:HEADER_WINDOW_LINES]
    return any("SPDX-License-Identifier: AGPL-3.0-or-later" in line for line in leading_lines)


def test_tracked_python_and_rust_sources_have_spdx_headers() -> None:
    """Owned sources carry headers; upstream sources retain REUSE licence metadata."""
    root = _repo_root()
    missing = [
        path.relative_to(root).as_posix()
        for path in _tracked_source_files(root)
        if spdx_header_audit.classify_path(root, path.relative_to(root).as_posix()).kind
        is not spdx_header_audit.CoverageKind.REUSE_METADATA
        and not _has_spdx_header(path)
    ]

    assert missing == []


def _tracked(root: Path, pathspec: str) -> set[str]:
    """Return the tracked repository paths that match one git pathspec."""
    completed = subprocess.run(
        ["git", "ls-files", pathspec], cwd=root, check=True, capture_output=True, text=True
    )
    return {line for line in completed.stdout.splitlines() if line}


def test_tracked_fixture_json_files_carry_their_ownership_sidecar() -> None:
    """Every fixture JSON has a tracked ``.license`` sidecar with the licence and both copyright records.

    JSON has no comment syntax, so its licence and ownership live in the adjacent sidecar. The
    ``tests/**`` annotation in ``REUSE.toml`` gives a licence but not the two copyright records,
    so it does not stand in for a sidecar.
    """
    root = _repo_root()
    fixtures = _tracked(root, FIXTURE_JSON_PATHSPEC)
    assert fixtures, "no tracked fixture JSON found"
    assert _sidecar_problems(root, fixtures) == []


def _sidecar_problems(root: Path, paths: set[str]) -> list[str]:
    """Return each JSON path without a tracked sidecar carrying the licence and both records."""
    sidecars = _tracked(root, "*.license")
    problems: list[str] = []
    for path in sorted(paths):
        sidecar = f"{path}.license"
        if sidecar not in sidecars:
            problems.append(f"{path}: no tracked sidecar")
            continue
        text = (root / sidecar).read_text(encoding="utf-8")
        problems.extend(
            f"{sidecar}: lacks {required!r}"
            for required in (SIDECAR_LICENCE, *SIDECAR_COPYRIGHT_RECORDS)
            if required not in text
        )
    return problems


def _legacy_json(root: Path) -> set[str]:
    """Return the paths the legacy list names, without its comment header."""
    lines = (root / LEGACY_JSON_LIST).read_text(encoding="utf-8").splitlines()
    return {line for line in lines if line and not line.startswith("#")}


def test_every_tracked_json_outside_the_legacy_list_carries_its_ownership_sidecar() -> None:
    """A JSON file added since the 2026-10-01 contract has its sidecar; vendored trees are exempt.

    Only the files that predate the contract may lack one, and only while the legacy list names
    them.
    """
    root = _repo_root()
    tracked = _tracked(root, "*.json") | _tracked(root, "*.JSON")
    legacy = _legacy_json(root)
    owned = {path for path in tracked if not path.startswith(VENDORED_PREFIXES)} - legacy
    assert owned, "no tracked JSON outside the legacy list"
    assert _sidecar_problems(root, owned) == []


def _json_added_since_the_contract(root: Path) -> set[str]:
    """Return every JSON path a commit added on or after the contract date, by the full history."""
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert shallow.stdout.strip() == "false", "the legacy check reads the whole history; fetch it"
    completed = subprocess.run(
        [
            "git",
            "log",
            "--diff-filter=A",
            f"--since={CONTRACT_DATE}",
            "--name-only",
            "--format=",
            "--",
            "*.json",
            "*.JSON",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line for line in completed.stdout.splitlines() if line}


def test_no_json_added_since_the_contract_is_on_the_legacy_list() -> None:
    """The list names only pre-contract files, and it is no longer than when it was written."""
    root = _repo_root()
    legacy = _legacy_json(root)
    assert sorted(legacy & _json_added_since_the_contract(root)) == []
    assert len(legacy) <= LEGACY_CEILING


def test_the_legacy_json_list_only_shrinks() -> None:
    """Every listed file is still tracked and still lacks a sidecar, so the list cannot hide a new one."""
    root = _repo_root()
    tracked = _tracked(root, "*.json") | _tracked(root, "*.JSON")
    sidecars = _tracked(root, "*.license")
    legacy = _legacy_json(root)
    assert legacy, "the legacy list is empty; delete it and its tests"
    untracked = sorted(path for path in legacy if path not in tracked)
    covered = sorted(path for path in legacy if f"{path}.license" in sidecars)
    assert untracked == [], f"remove from {LEGACY_JSON_LIST}: no longer tracked"
    assert covered == [], f"remove from {LEGACY_JSON_LIST}: they now carry a sidecar"
