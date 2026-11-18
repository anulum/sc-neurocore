# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SPDX header policy tests

"""Policy tests for tracked Python and Rust source headers and for fixture JSON ownership sidecars."""

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
    sidecars = _tracked(root, f"{FIXTURE_JSON_PATHSPEC}.license")
    assert fixtures, "no tracked fixture JSON found"
    problems: list[str] = []
    for path in sorted(fixtures):
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

    assert problems == []
