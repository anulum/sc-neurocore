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
import sys
from typing import TYPE_CHECKING, Any

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"
NAVIGATION_POLICY = REPO_ROOT / "docs" / "navigation_policy.toml"


class _MkDocsConfigLoader(yaml.SafeLoader):
    """Parse MkDocs metadata without importing extension callback objects."""


def _ignore_python_name(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> str:
    return tag_suffix


if not TYPE_CHECKING:
    _MkDocsConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _ignore_python_name)


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


def _pathspec_patterns(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise TypeError(f"Unsupported MkDocs pathspec type: {type(raw)!r}")


def test_unlisted_docs_are_classified_by_navigation_policy() -> None:
    """Every public doc outside MkDocs nav must have an explicit policy bucket."""
    config = yaml.load(MKDOCS_CONFIG.read_text(encoding="utf-8"), Loader=_MkDocsConfigLoader)
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


def test_classified_unlisted_docs_are_silenced_in_mkdocs_config() -> None:
    """Classified unlisted docs should not emit MkDocs nav-omission INFO noise."""
    config = yaml.load(MKDOCS_CONFIG.read_text(encoding="utf-8"), Loader=_MkDocsConfigLoader)
    listed = _nav_paths(config["nav"])
    unlisted = _public_markdown_paths() - listed
    not_in_nav = _pathspec_patterns(config.get("not_in_nav"))

    assert not_in_nav
    unsilenced = [
        doc_path
        for doc_path in sorted(unlisted)
        if not any(fnmatch(doc_path, pattern) for pattern in not_in_nav)
    ]

    assert not unsilenced
