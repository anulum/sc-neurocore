# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One changelog, copied byte for byte into the documentation

"""Keep the documentation changelog identical to the repository changelog.

The documentation site and the release workflow (``release.yml``) read ``docs/CHANGELOG.md``.
Edited separately from ``CHANGELOG.md``, the two files had diverged by hundreds of entries.
Entries are written in ``CHANGELOG.md`` only and copied over. Each unreleased entry
sits under one of the standard categories.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_changelog_is_the_repository_changelog() -> None:
    """``docs/CHANGELOG.md`` is a byte copy of ``CHANGELOG.md``."""
    canonical = (ROOT / "CHANGELOG.md").read_bytes()
    assert (ROOT / "docs" / "CHANGELOG.md").read_bytes() == canonical, (
        "docs/CHANGELOG.md differs from CHANGELOG.md; edit CHANGELOG.md and run "
        "`cp CHANGELOG.md docs/CHANGELOG.md`"
    )


def test_every_changelog_entry_is_written_once() -> None:
    """No entry appears twice, so a merge or a copy cannot duplicate history silently."""
    entries: list[str] = []
    current: list[str] | None = None
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            if current is not None:
                entries.append(" ".join(" ".join(current).split()))
            current = [line]
        elif current is not None and line.startswith("  "):
            current.append(line)
        else:
            if current is not None:
                entries.append(" ".join(" ".join(current).split()))
            current = None
    if current is not None:
        entries.append(" ".join(" ".join(current).split()))
    repeated = sorted({entry[:120] for entry in entries if entries.count(entry) > 1})
    assert entries and not repeated, repeated


CATEGORIES = ("Security", "Added", "Changed", "Fixed", "Removed", "Repository hygiene")


def _unreleased_lines() -> list[str]:
    """Return the lines of the ``[Unreleased]`` section of ``CHANGELOG.md``.

    Returns
    -------
    list[str]
        The lines after the section's heading, up to the next release heading.
    """
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = "## [Unreleased]\n"
    start = text.index(heading) + len(heading)
    end = text.find("\n## [", start)
    return text[start : end if end >= 0 else len(text)].splitlines()


def test_unreleased_entries_sit_under_the_standard_categories() -> None:
    """Every unreleased entry sits under one standard category, in the standard order.

    The categories are Security, Added, Changed, Fixed, Removed and Repository
    hygiene. Each heading appears at most once and holds at least one entry, and
    no entry comes before the first heading.
    """
    headings: list[str] = []
    entries: dict[str, int] = {}
    stray: list[str] = []
    for line in _unreleased_lines():
        if line.startswith("### "):
            headings.append(line[4:].strip())
            entries.setdefault(headings[-1], 0)
        elif line.startswith("- "):
            if headings:
                entries[headings[-1]] += 1
            else:
                stray.append(line[:80])
    unknown = [name for name in headings if name not in CATEGORIES]
    assert not stray, f"entries before any category heading: {stray}"
    assert not unknown, f"headings other than {CATEGORIES}: {unknown}"
    in_order = sorted(set(headings), key=CATEGORIES.index)
    assert headings == in_order, f"headings repeated or out of order: {headings}"
    empty = [name for name, count in entries.items() if count == 0]
    assert not empty, f"categories without an entry: {empty}"
