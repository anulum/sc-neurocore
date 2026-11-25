# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Notebooks and documentation name only the Python versions the package supports

"""Keep every supported-Python statement within the package floor and the tested interpreters.

The floor is ``requires-python`` in ``pyproject.toml``. The tested interpreters are the
``python-minor`` matrix of the ``test`` job in ``.github/workflows/ci.yml``. No tracked
notebook's kernel metadata may name an interpreter below the floor. No statement such as
``Python 3.11+``, ``Python >= 3.11``, ``Python 3.11 to 3.14`` or ``supports Python 3.11`` in a
tracked Markdown page, or in a docstring of the tracked package, tool, script or bridge sources,
may start below the floor or end above the newest tested interpreter.
Changelogs, release notes, dated reports and the version 3 migration guide are historical
records and keep their original facts.
"""

from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_FILES = frozenset(
    {"CHANGELOG.md", "docs/CHANGELOG.md", "docs/development/v3_migration.md"}
)
HISTORICAL_PREFIXES = ("docs/development/RELEASE_NOTES_", "docs/reports/")
STATEMENTS = (
    re.compile(r"Python\s*\**\s*3\.(?P<low>\d+)\s*(?:–|—|-|to|through)\s*3\.(?P<high>\d+)"),
    re.compile(
        r"Python\s*(?:>=\s*)?\**\s*3\.(?P<low>\d+)\s*\**\s*"
        r"(?:\+|or newer|or later|and newer|and later)"
    ),
    re.compile(r"Python\s*>=\s*3\.(?P<low>\d+)"),
    re.compile(r"\bsupports?\s+Python\s+3\.(?P<low>\d+)\b"),
)
COUNTS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _floor() -> int:
    """Return the minor version of the package's ``requires-python`` floor.

    Returns
    -------
    int
        The minor version ``N`` of ``>=3.N``.
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    match = re.fullmatch(r">=\s*3\.(\d+)", cast(str, project["requires-python"]))
    assert match is not None, project["requires-python"]
    return int(match.group(1))


def _tested() -> list[int]:
    """Return the minor versions the CI test job runs.

    Returns
    -------
    list of int
        The ``python-minor`` matrix of the ``test`` job, in order.
    """
    workflow = cast(
        dict[str, Any],
        yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")),
    )
    minors = cast(list[str], workflow["jobs"]["test"]["strategy"]["matrix"]["python-minor"])
    return [int(minor.removeprefix("3.")) for minor in minors]


def _tracked(*patterns: str) -> list[str]:
    """Return the tracked files that match ``patterns``.

    Parameters
    ----------
    *patterns : str
        Git pathspecs.

    Returns
    -------
    list of str
        Repository-relative paths, sorted.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *patterns],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(name for name in listed.split("\0") if name)


def _current_pages() -> list[str]:
    """Return the tracked Markdown pages that describe the current package.

    Returns
    -------
    list of str
        Every tracked ``*.md`` path except the historical records.
    """
    return [
        name
        for name in _tracked("*.md")
        if name not in HISTORICAL_FILES and not name.startswith(HISTORICAL_PREFIXES)
    ]


def _docstrings(name: str) -> list[tuple[int, str]]:
    """Return the module, class and function docstrings of one Python source.

    Parameters
    ----------
    name : str
        Repository-relative path of the source.

    Returns
    -------
    list of tuple
        ``(line of the documented node, docstring)`` for each docstring.
    """
    tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return [
        (getattr(node, "lineno", 1), docstring)
        for node in ast.walk(tree)
        if isinstance(node, documented)
        and (docstring := ast.get_docstring(node, clean=False)) is not None
    ]


def _statements(text: str) -> list[tuple[int, str, int, int | None]]:
    """Find the supported-Python statements of one page.

    Parameters
    ----------
    text : str
        The page.

    Returns
    -------
    list of tuple
        ``(line, statement, lowest minor, highest minor or None)`` for each statement.
    """
    found = []
    for pattern in STATEMENTS:
        for match in pattern.finditer(text):
            high = match.groupdict().get("high")
            found.append(
                (
                    text.count("\n", 0, match.start()) + 1,
                    " ".join(match.group(0).split()),
                    int(match.group("low")),
                    None if high is None else int(high),
                )
            )
    return found


def test_the_ci_test_matrix_starts_at_the_package_floor() -> None:
    """CI tests the floor itself, and each interpreter it tests is at or above the floor."""
    floor = _floor()
    tested = _tested()
    assert floor >= 11
    assert tested == sorted(tested)
    assert tested[0] == floor


def test_no_notebook_names_an_interpreter_below_the_floor() -> None:
    """A notebook's kernel metadata names no Python older than ``requires-python`` allows."""
    floor = _floor()
    stale = []
    notebooks = _tracked("*.ipynb")
    assert notebooks
    for name in notebooks:
        metadata = cast(
            dict[str, Any], json.loads((ROOT / name).read_text(encoding="utf-8"))["metadata"]
        )
        version = cast(dict[str, Any], metadata.get("language_info", {})).get("version")
        if version is not None:
            major, minor = (int(part) for part in str(version).split(".")[:2])
            if (major, minor) < (3, floor):
                stale.append(f"{name}: language_info.version {version}")
        display = str(cast(dict[str, Any], metadata.get("kernelspec", {})).get("display_name", ""))
        named = re.search(r"Python\s+3\.(\d+)", display)
        if named is not None and int(named.group(1)) < floor:
            stale.append(f"{name}: kernelspec.display_name {display}")
    assert stale == []


def test_no_current_page_promises_an_unsupported_or_untested_python() -> None:
    """Every stated range starts at or above the floor and ends at or below the newest tested."""
    floor = _floor()
    newest = _tested()[-1]
    wrong = []
    for name in _current_pages():
        text = (ROOT / name).read_text(encoding="utf-8")
        for line, statement, low, high in _statements(text):
            if low < floor or (high is not None and not low <= high <= newest):
                wrong.append(f"{name}:{line}: {statement}")
    assert wrong == []


def test_no_docstring_promises_an_unsupported_or_untested_python() -> None:
    """Docstrings of the package, tools, scripts and bridge obey the same bounds as the pages."""
    floor = _floor()
    newest = _tested()[-1]
    wrong = []
    for name in _tracked("src/*.py", "tools/*.py", "scripts/*.py", "bridge/*.py"):
        for node_line, docstring in _docstrings(name):
            for _, statement, low, high in _statements(docstring):
                if low < floor or (high is not None and not low <= high <= newest):
                    wrong.append(f"{name}:{node_line}: {statement}")
    assert wrong == []


def test_the_faq_answer_is_the_tested_matrix() -> None:
    """The FAQ names the floor, the newest tested interpreter and how many CI tests."""
    tested = _tested()
    faq = (ROOT / "docs/guides/faq.md").read_text(encoding="utf-8")
    answer = faq.split("### Which Python versions are supported?", 1)[1].split("\n### ", 1)[0]
    assert f"Python 3.{tested[0]}–3.{tested[-1]}." in answer
    assert f"CI tests all {COUNTS[len(tested)]} versions" in answer


def test_the_statement_finder_reads_each_form() -> None:
    """Each statement form yields its bounds, and a third-party version note yields none."""
    page = (
        "Python 3.11+, Python >= 3.12, Python **3.13 or newer**, Python 3.11 to 3.14,\n"
        "Python 3.11–3.13, supports Python 3.12; lava-nc runs on Python 3.10 only."
    )
    assert Counter((low, high) for _, _, low, high in _statements(page)) == Counter(
        [(11, None), (12, None), (13, None), (11, 14), (11, 13), (12, None)]
    )
