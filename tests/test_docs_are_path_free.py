# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A path under someone's home or removable media, written absolutely. Matched
#: by shape rather than by one operator's name: the next author's machine is a
#: different string and the same defect.
_MACHINE_PATH = re.compile(r"(?:/home/[a-z][a-z0-9_-]*|/media/[a-z][a-z0-9_-]*|/Users/[A-Za-z])/")

#: Conventional stand-ins a reader is expected to replace. These are the remedy
#: for a machine-specific path, not an instance of one, so the shape rule above
#: must not object to them. `<your-home>` and friends never match it; `user` and
#: `you` are ordinary account names and do.
_PLACEHOLDER_ACCOUNTS = frozenset({"user", "username", "you", "youruser", "me"})


def _is_placeholder(line: str) -> bool:
    """Return whether every machine-shaped path on the line is a stand-in."""
    matches = _MACHINE_PATH.findall(line) or [
        match.group(0) for match in _MACHINE_PATH.finditer(line)
    ]
    accounts = [segment.strip("/").split("/")[-1] for segment in matches]
    return bool(accounts) and all(account in _PLACEHOLDER_ACCOUNTS for account in accounts)


#: `docs/internal/` is private by policy and legitimately records live operator
#: paths; it is never published and is excluded here for that reason.
_PRIVATE = "internal"


def _published_pages() -> list[Path]:
    """Return every Markdown page under ``docs/`` that is published."""
    return sorted(
        path
        for path in (REPO_ROOT / "docs").rglob("*.md")
        if _PRIVATE not in path.relative_to(REPO_ROOT / "docs").parts
    )


def test_published_documentation_names_no_machine_specific_path() -> None:
    """A reader must be able to run what a page shows on their own machine.

    Five pages carried the maintainer's own absolute paths — a `CARGO_TARGET_DIR`
    under a personal scratch directory inside a build recipe, captured tool
    output naming a home directory, and a collection root that disclosed the
    private coordination layout. Each is unrunnable for anyone else, and the last
    published structure that is not meant to be public.
    """
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()[:90]}"
        for path in _published_pages()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _MACHINE_PATH.search(line) and not _is_placeholder(line)
    ]

    assert offenders == [], "published pages naming a machine-specific path:\n" + "\n".join(
        offenders
    )
