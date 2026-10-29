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
Entries are written in ``CHANGELOG.md`` only and copied over.
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
