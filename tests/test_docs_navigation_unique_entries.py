# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Each documentation page and section appears once in the site navigation

"""Keep the MkDocs navigation free of repeated sections and pages.

MkDocs accepts a navigation list in which two sibling sections share a title, or one page is
listed in two places. The rendered site then shows two tabs with the same name, or a page whose
previous and next links follow only one of its places. Every section title is unique among its
siblings, and every page is listed once.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, cast

import yaml

MKDOCS_CONFIG = Path(__file__).resolve().parents[1] / "mkdocs.yml"


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


def _navigation() -> list[Any]:
    """Return the ``nav`` list of ``mkdocs.yml``.

    Returns
    -------
    list
        The navigation entries.
    """
    config = yaml.load(MKDOCS_CONFIG.read_text(encoding="utf-8"), Loader=_MKDOCS_LOADER)
    return cast(list[Any], cast(dict[str, Any], config)["nav"])


def _repeated_titles(entries: list[Any], trail: tuple[str, ...]) -> list[str]:
    """Find titles that two sibling entries share, at every level.

    Parameters
    ----------
    entries : list
        One level of the navigation.
    trail : tuple of str
        The section titles above this level.

    Returns
    -------
    list of str
        ``"Section > Title (count)"`` for each repeated sibling title.
    """
    titles: Counter[str] = Counter()
    repeated = []
    for entry in entries:
        if isinstance(entry, dict):
            for title, value in cast(dict[str, Any], entry).items():
                titles[title] += 1
                if isinstance(value, list):
                    repeated.extend(_repeated_titles(value, (*trail, title)))
    for title, count in titles.items():
        if count > 1:
            repeated.append(f"{' > '.join(trail) or '(top)'} > {title} ({count})")
    return repeated


def _pages(entries: list[Any]) -> list[str]:
    """List every page path of the navigation, once per place it is listed.

    Parameters
    ----------
    entries : list
        One level of the navigation.

    Returns
    -------
    list of str
        The page paths in navigation order.
    """
    pages = []
    for entry in entries:
        if isinstance(entry, str):
            pages.append(entry)
        elif isinstance(entry, dict):
            for value in cast(dict[str, Any], entry).values():
                if isinstance(value, str):
                    pages.append(value)
                elif isinstance(value, list):
                    pages.extend(_pages(value))
    return pages


def test_no_two_sibling_sections_or_pages_share_a_title() -> None:
    """At every level of the navigation, sibling titles are distinct."""
    assert _repeated_titles(_navigation(), ()) == []


def test_every_page_is_listed_once() -> None:
    """No page path appears in two places of the navigation."""
    counts = Counter(_pages(_navigation()))
    assert sorted(page for page, count in counts.items() if count > 1) == []


def test_the_checks_see_a_repeated_title_and_a_repeated_page() -> None:
    """A navigation with both defects is reported, at the level where each occurs."""
    navigation = [
        {"Benchmarks": [{"A": "a.md"}]},
        {"Guides": [{"B": "b.md"}, {"B": "c.md"}, "a.md"]},
        {"Benchmarks": [{"D": "d.md"}]},
    ]
    assert sorted(_repeated_titles(navigation, ())) == [
        "(top) > Benchmarks (2)",
        "Guides > B (2)",
    ]
    assert Counter(_pages(navigation))["a.md"] == 2
