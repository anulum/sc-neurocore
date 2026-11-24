# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Group and project map drift checks

"""Bind the group and project map (docs/ecosystem.md) to the split map and to its own counts.

The page states how many programme entries exist, which repositories they are, and where each planned
SC-NC destination's code lives today. These checks fail when the page drifts from
``tools/sc_nc_split_map.json`` or from the rows it actually lists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "ecosystem.md"
SPLIT_MAP = ROOT / "tools" / "sc_nc_split_map.json"
SOURCE = "https://github.com/anulum/sc-neurocore/"
ROW = re.compile(
    r'^\| <a id="(?P<anchor>[A-Za-z0-9-]+)"></a>\[\*\*(?P<name>[A-Z0-9-]+)\*\*\]'
    r"\((?P<link>https://github\.com/anulum/[A-Za-z0-9-]+)\)(?P<rest>.*)$",
    re.MULTILINE,
)
ENTRY = re.compile(r"/(?:tree|blob)/main/src/sc_neurocore/(?P<name>[A-Za-z0-9_]+?)(?:\.py)?\)")


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _split() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(SPLIT_MAP.read_text(encoding="utf-8"))
    return loaded


def _section(text: str, title: str) -> str:
    """Return the body of the level-two section ``title``, up to the next level-two heading."""
    start = text.index(f"\n## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


def _rows(section: str) -> list[re.Match[str]]:
    return list(ROW.finditer(section))


def _words(items: list[str]) -> str:
    quoted = [f"`{item}`" for item in items]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_the_stated_entry_counts_match_the_listed_rows() -> None:
    """The "N entries: 1 + a + b + c" sentence equals the anchored rows of each section."""
    text = _page()
    stated = re.search(
        r"\*\*(\d+) entries: 1 workbench \+ (\d+) neuromorphic destinations \+\s+"
        r"(\d+) measurement and evidence projects \+ (\d+) circuit destinations\*\*",
        text,
    )
    assert stated is not None
    total, neuromorphic, evidence, circuit = (int(group) for group in stated.groups())
    assert len(_rows(_section(text, "Specialised repositories"))) == neuromorphic
    assert len(_rows(_section(text, "Supporting tools"))) == evidence
    assert len(_rows(_section(text, "General circuit-design platform"))) == circuit
    assert '<a id="sc-neurocore"></a>[SC-NEUROCORE](https://github.com/anulum/sc-neurocore)' in text
    assert total == 1 + neuromorphic + evidence + circuit


def test_every_listed_repository_links_its_lower_case_name() -> None:
    """Each anchored row links github.com/anulum/<name in lower case> under that same anchor."""
    rows = _rows(_page())
    assert rows
    for row in rows:
        assert row["anchor"] == row["name"].lower(), row["name"]
        assert row["link"] == f"https://github.com/anulum/{row['name'].lower()}", row["name"]
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names)), "a repository is listed twice"


def test_the_specialised_rows_are_the_split_map_destinations() -> None:
    """The thirteen SC-NC rows name exactly the destinations of the split map."""
    rows = _rows(_section(_page(), "Specialised repositories"))
    assert sorted(row["name"] for row in rows) == sorted(_split()["order"])


def test_where_it_lives_today_lists_exactly_the_split_map_entries() -> None:
    """Each SC-NC row's last cell links every package and module the map assigns to it, and no other."""
    split = _split()
    expected: dict[str, set[str]] = {destination: set() for destination in split["order"]}
    for name, destination in [*split["packages"].items(), *split["modules"].items()]:
        expected[destination].add(name)
    for row in _rows(_section(_page(), "Specialised repositories")):
        where = row["rest"].rstrip(" |").rsplit(" | ", 1)[-1]
        listed = [entry["name"] for entry in ENTRY.finditer(where)]
        assert len(listed) == len(set(listed)), f"{row['name']} lists an entry twice"
        assert set(listed) == expected[row["name"]], row["name"]


def test_the_import_order_sentence_follows_the_split_map() -> None:
    """The page states the destinations' import order exactly as the map orders them."""
    order = _split()["order"]
    sentence = (
        "Their imports follow one order: " + ", ".join(order[:-1]) + " and " + order[-1] + "."
    )
    assert sentence in _compact(_page())


def test_the_facade_and_circuit_side_sentences_follow_the_split_map() -> None:
    """The page names the facade that stays and the packages that leave for the circuit platform."""
    split = _split()
    text = _compact(_page())
    facade_modules = [f"{name}.py" for name in split["facade_modules"]]
    assert (
        f"the {_words(split['facade_packages'])} packages and the {_words(facade_modules)} modules."
        in text
    )
    moves = sorted(split["circuit_side_packages"].items())
    links = [f"`{package}` to [{target}](#{target.lower()})" for package, target in moves]
    assert ", ".join(links[:-1]) + " and " + links[-1] + "." in text
    for _, target in moves:
        assert f'<a id="{target.lower()}"></a>' in text


def test_the_page_links_the_split_map_and_its_gate() -> None:
    """The plan the cells come from, and the gate that measures it, are both linked."""
    text = _page()
    assert f"({SOURCE}blob/main/tools/sc_nc_split_map.json)" in text
    assert "(development/maintenance_tools.md#split-import-gate)" in text
