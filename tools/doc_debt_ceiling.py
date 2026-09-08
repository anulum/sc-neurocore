#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One ratchet rule shared by every language's documentation debt

"""The committed-ceiling machinery a per-language documentation ratchet needs.

The owner directive of 2026-09-06 asks each language to enforce its debt "so it
cannot grow", with the ceiling set at the measured figure. The measuring is
language-specific and belongs in a tool that understands the language; the rule
about the ceiling is not, and it is the half that must not drift.

The rule is one sentence: **the ceiling may fall and may not rise.** A tool
lowers it; raising it is an edit somebody makes on purpose, in a diff a reviewer
can see. When two languages each carried their own copy of that sentence, the
copies were free to disagree about it, and a reader could not tell which one the
project meant. They share this module instead.

A ceiling that cannot be read stops the check. It never defaults to a permissive
figure: a ratchet that passes when it cannot see the ceiling enforces nothing
while looking exactly like a ratchet that does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RatchetError(RuntimeError):
    """Raised when no verdict is possible, so the check must stop."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """The comparison of one measurement against the committed ceiling.

    Attributes
    ----------
    language : str
        The language the figure covers, as it is named to a reader.
    measured : int
        Undocumented declarations the measurement reported.
    ceiling : int
        The committed ceiling it was compared against.
    ok : bool
        Whether the measurement is at or below the ceiling.
    """

    language: str
    measured: int
    ceiling: int
    ok: bool

    def summary(self) -> str:
        """Return the one line a reader of the output needs."""
        if self.measured > self.ceiling:
            return (
                f"{self.language} documentation debt rose: {self.measured} undocumented "
                f"items, ceiling {self.ceiling} (+{self.measured - self.ceiling}). "
                "Document the new items, or lower nothing and raise the ceiling "
                "deliberately in a reviewed diff."
            )
        if self.measured < self.ceiling:
            return (
                f"{self.language} documentation debt fell: {self.measured} undocumented "
                f"items, ceiling {self.ceiling} (-{self.ceiling - self.measured}). "
                "Run with --update to lower the ceiling."
            )
        return (
            f"{self.language} documentation debt unchanged at {self.measured} undocumented items."
        )


def compare(measured: int, ceiling: int, *, language: str) -> Verdict:
    """Return the verdict for one measurement against the ceiling."""
    return Verdict(language=language, measured=measured, ceiling=ceiling, ok=measured <= ceiling)


def read_ceiling(path: Path) -> int:
    """Return the committed ceiling.

    Raises
    ------
    RatchetError
        The record is absent or does not carry a usable ceiling; a missing
        ceiling must stop the check rather than default to a permissive one.
    """
    if not path.is_file():
        raise RatchetError(f"no ceiling record at {path}")
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    ceiling = document.get("undocumented")
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 0:
        raise RatchetError("the ceiling record carries no whole-number ceiling")
    return ceiling


def write_ceiling(
    path: Path,
    *,
    undocumented: int,
    files: int,
    note: str,
    schema_version: str,
    provenance: dict[str, str],
) -> None:
    """Write a ceiling record, refusing to raise an existing one.

    Parameters
    ----------
    path : Path
        Where the record lives.
    undocumented : int
        The measured figure that becomes the new ceiling.
    files : int
        How many files carry at least one of those declarations.
    note : str
        What the record means, written for whoever opens the file first.
    schema_version : str
        The contract version of this record.
    provenance : dict of str to str
        Tool version, argv and source digest, so the figure can be re-taken.

    Raises
    ------
    RatchetError
        The new figure is above the recorded one. Lowering is the tool's job;
        raising is a decision, and it belongs in a diff somebody signed.
    """
    if path.is_file():
        current = read_ceiling(path)
        if undocumented > current:
            raise RatchetError(
                f"refusing to raise the ceiling from {current} to {undocumented}; "
                "a rise is a deliberate edit, not an automatic one"
            )
    document = {
        "note": note,
        "provenance": dict(sorted(provenance.items())),
        "schema_version": schema_version,
        "undocumented": undocumented,
        "undocumented_files": files,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
