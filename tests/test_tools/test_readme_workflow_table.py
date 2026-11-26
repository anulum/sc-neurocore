# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The README workflow count and the CI page table list the workflows that exist

"""Bind the README's workflow count and the CI/CD page's table to the workflow files."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def test_readme_and_ci_page_count_and_list_every_workflow_file() -> None:
    """The README and CI/CD page counts equal the number of files; each file has one row."""
    files = sorted(path.name for path in (ROOT / ".github/workflows").glob("*.yml"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs/development/ci_workflows.md").read_text(encoding="utf-8")
    assert files, "the repository has workflow files"
    assert f"| GitHub Actions workflows | {len(files)} |" in readme
    assert f"\n{len(files)} GitHub Actions workflows (`.github/workflows/`)" in page
    listed = re.findall(r"^\| \*\*([a-z0-9-]+\.yml)\*\* \|", page, flags=re.MULTILINE)
    assert sorted(listed) == files
