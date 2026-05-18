# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — documentation navigation policy tests

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import tomllib
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"
NAVIGATION_POLICY = REPO_ROOT / "docs" / "navigation_policy.toml"


def _nav_paths(nav: list[Any]) -> set[str]:
    paths: set[str] = set()

    for item in nav:
        if isinstance(item, str):
            paths.add(item)
            continue
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    paths.add(value)
                elif isinstance(value, list):
                    paths.update(_nav_paths(value))
    return paths


def _public_markdown_paths() -> set[str]:
    return {
        path.relative_to(DOCS_ROOT).as_posix()
        for path in DOCS_ROOT.rglob("*.md")
        if "internal" not in path.relative_to(DOCS_ROOT).parts
    }


def test_unlisted_docs_are_classified_by_navigation_policy() -> None:
    """Every public doc outside MkDocs nav must have an explicit policy bucket."""
    config = yaml.load(MKDOCS_CONFIG.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)
    listed = _nav_paths(config["nav"])
    unlisted = _public_markdown_paths() - listed

    policy = tomllib.loads(NAVIGATION_POLICY.read_text(encoding="utf-8"))
    classifications = policy["classification"]
    matched: dict[str, list[str]] = {entry["name"]: [] for entry in classifications}
    unclassified: list[str] = []

    for doc_path in sorted(unlisted):
        matches = [
            entry["name"]
            for entry in classifications
            if any(fnmatch(doc_path, pattern) for pattern in entry["patterns"])
        ]
        if len(matches) == 1:
            matched[matches[0]].append(doc_path)
        else:
            unclassified.append(doc_path)

    assert not unclassified
    for entry in classifications:
        assert len(matched[entry["name"]]) == entry["expected_count"], entry["name"]
