# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The site navigation is organised by what a reader came to do

"""Keep the MkDocs navigation in the Diátaxis arrangement.

The top level is Home, then the four Diátaxis kinds (Tutorials for learning, How-to Guides
for tasks, Reference for lookup and Explanation for understanding), then Evidence for the
measured results, the experimental Studio and the Project records. Pages whose directory
already says what they are stay in their section: tutorials under Tutorials; validation,
benchmark and report pages under Evidence; architecture and research pages under
Explanation; development records under Project; Studio pages under the Studio; and API
pages under Reference, apart from the Studio's own OpenAPI contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

MKDOCS_CONFIG = Path(__file__).resolve().parents[1] / "mkdocs.yml"
SECTIONS = (
    "Home",
    "Tutorials",
    "How-to Guides",
    "Reference",
    "Explanation",
    "Evidence",
    "Studio (Experimental)",
    "Project",
)
HOMES = {
    "tutorials": {"Tutorials"},
    "validation": {"Evidence"},
    "benchmarks": {"Evidence"},
    "reports": {"Evidence"},
    "architecture": {"Explanation"},
    "research": {"Explanation"},
    "development": {"Project"},
    "studio": {"Studio (Experimental)"},
    "api": {"Reference", "Studio (Experimental)"},
}


def _python_name(loader: Any, tag_suffix: str, node: Any) -> str:
    """Return a ``!!python/name:`` tag as its dotted name instead of importing it.

    Parameters
    ----------
    loader : yaml.SafeLoader
        The active loader.
    tag_suffix : str
        The dotted name after the tag prefix.
    node : yaml.Node
        The tagged node.

    Returns
    -------
    str
        The dotted name.
    """
    return tag_suffix


# A safe loader of its own, so the extra constructor never reaches yaml.SafeLoader itself.
_MKDOCS_LOADER: Any = type("_MkDocsConfigLoader", (yaml.SafeLoader,), {})
_MKDOCS_LOADER.add_multi_constructor("tag:yaml.org,2002:python/name:", _python_name)


def _navigation() -> list[dict[str, Any]]:
    """Return the top-level entries of the ``nav`` list of ``mkdocs.yml``.

    Returns
    -------
    list of dict
        One single-key mapping per top-level entry.
    """
    config = yaml.load(MKDOCS_CONFIG.read_text(encoding="utf-8"), Loader=_MKDOCS_LOADER)
    return cast(list[dict[str, Any]], cast(dict[str, Any], config)["nav"])


def _pages(value: Any) -> list[str]:
    """List the page paths under one navigation entry's value.

    Parameters
    ----------
    value : str or list
        A page path, or a list of nested entries.

    Returns
    -------
    list of str
        The page paths, in navigation order.
    """
    if isinstance(value, str):
        return [value]
    pages = []
    for entry in value:
        if isinstance(entry, str):
            pages.append(entry)
        else:
            for nested in cast(dict[str, Any], entry).values():
                pages.extend(_pages(nested))
    return pages


def _misplaced(navigation: list[dict[str, Any]]) -> list[str]:
    """Find pages listed outside the section their directory belongs to.

    Parameters
    ----------
    navigation : list of dict
        The top-level navigation entries.

    Returns
    -------
    list of str
        ``"section: page"`` for each misplaced page.
    """
    misplaced = []
    for entry in navigation:
        for section, value in entry.items():
            for page in _pages(value):
                homes = HOMES.get(page.split("/", 1)[0]) if "/" in page else None
                if homes is not None and section not in homes:
                    misplaced.append(f"{section}: {page}")
    return misplaced


def test_the_top_level_is_the_diataxis_arrangement() -> None:
    """The top-level sections are exactly the eight sections, in reading order."""
    assert tuple(next(iter(entry)) for entry in _navigation()) == SECTIONS


def test_each_page_with_a_home_directory_stays_in_its_section() -> None:
    """Tutorials, evidence, explanation, project, Studio and API pages stay in their sections."""
    assert _misplaced(_navigation()) == []


def test_the_placement_check_sees_a_misplaced_page() -> None:
    """A validation page under Reference, or a tutorial under How-to Guides, is reported."""
    navigation: list[dict[str, Any]] = [
        {"Reference": [{"Fidelity": "validation/x.md"}, {"API": "api/x.md"}]},
        {"How-to Guides": [{"Group": [{"First steps": "tutorials/01.md"}]}, "guides/y.md"]},
    ]
    assert _misplaced(navigation) == [
        "Reference: validation/x.md",
        "How-to Guides: tutorials/01.md",
    ]
