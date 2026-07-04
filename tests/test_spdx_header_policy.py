# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SPDX header policy tests

"""Policy tests for tracked Python and Rust source headers."""

from __future__ import annotations

import subprocess
from pathlib import Path


SOURCE_EXTENSIONS = (".py", ".rs")
"""Tracked source extensions covered by the S001/S002 audit items."""

HEADER_WINDOW_LINES = 10
"""Number of leading lines allowed for shebangs and policy header comments."""


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
    """Every tracked Python and Rust source file carries an SPDX header."""
    root = _repo_root()
    missing = [
        path.relative_to(root).as_posix()
        for path in _tracked_source_files(root)
        if not _has_spdx_header(path)
    ]

    assert missing == []
