# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The README workflow table lists the workflows that exist

"""Bind the README's workflow count and table to the workflow files."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def test_readme_counts_and_lists_every_workflow_file() -> None:
    """Both counts in the README equal the number of files, and each file has one row."""
    files = sorted(path.name for path in (ROOT / ".github/workflows").glob("*.yml"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert files, "the repository has workflow files"
    assert f"| GitHub Actions workflows | {len(files)} |" in readme
    assert f"\n{len(files)} GitHub Actions workflows (`.github/workflows/`)" in readme
    section = readme.split("## CI/CD", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(r"^\| \*\*([a-z0-9-]+\.yml)\*\* \|", section, flags=re.MULTILINE)
    assert sorted(listed) == files
