#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust documentation debt may fall, never rise

"""A ratchet on the engine crate's undocumented items.

The owner directive of 2026-09-06 requires that documentation debt be enforced
so it *cannot grow*, with the ratchet set at the measured ceiling. The obvious
way to do that in Rust — ``#![warn(missing_docs)]`` at the crate root — is not
available here, and the reason is worth stating rather than discovering: CI
runs ``cargo clippy -- -D warnings`` on this crate, so a crate-level warning
lint would turn 3873 warnings into 3873 errors and break every seat's build on
the first commit.

So the lint is run deliberately, by this tool, and compared against a committed
ceiling. Debt may fall and the ceiling follows it down; debt may not rise, and
the ceiling cannot be raised by the tool at all. Raising it is an edit someone
has to make on purpose, in a diff a reviewer can see.

The directive's other half — *deny at zero* — is served per module: a module
whose debt reaches zero declares ``#![deny(missing_docs)]`` and is closed
permanently, rather than being held open by a number somebody has to keep
re-checking.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.documentation_debt import read_rustc_stderr

#: Contract version of the ceiling record.
RUST_DOC_CEILING_SCHEMA_VERSION = "sc-neurocore.rust-doc-ceiling.v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Where the committed ceiling lives, beside the crate it constrains.
DEFAULT_CEILING = REPO_ROOT / "engine" / "missing_docs_ceiling.json"


class RatchetError(RuntimeError):
    """Raised when the lint cannot be run, so no verdict is possible."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """The comparison of a measurement against the committed ceiling.

    Attributes
    ----------
    measured : int
        Undocumented items the lint reported.
    ceiling : int
        The committed ceiling it was compared against.
    ok : bool
        Whether the measurement is at or below the ceiling.
    """

    measured: int
    ceiling: int
    ok: bool

    def summary(self) -> str:
        """Return the one line a reader of CI output needs."""
        if self.measured > self.ceiling:
            return (
                f"Rust documentation debt rose: {self.measured} undocumented items, "
                f"ceiling {self.ceiling} (+{self.measured - self.ceiling}). "
                "Document the new items, or lower nothing and raise the ceiling "
                "deliberately in a reviewed diff."
            )
        if self.measured < self.ceiling:
            return (
                f"Rust documentation debt fell: {self.measured} undocumented items, "
                f"ceiling {self.ceiling} (-{self.ceiling - self.measured}). "
                "Run with --update to lower the ceiling."
            )
        return f"Rust documentation debt unchanged at {self.measured} undocumented items."


def compare(measured: int, ceiling: int) -> Verdict:
    """Return the verdict for one measurement against the ceiling."""
    return Verdict(measured=measured, ceiling=ceiling, ok=measured <= ceiling)


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
    if not isinstance(ceiling, int) or ceiling < 0:
        raise RatchetError("the ceiling record carries no whole-number ceiling")
    return ceiling


def write_ceiling(path: Path, *, undocumented: int, files: int, provenance: dict[str, str]) -> None:
    """Write a ceiling record, refusing to raise an existing one.

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
        "note": (
            "Undocumented items in the engine crate, measured with rustc's own "
            "missing_docs lint. This ceiling may fall and may not rise: see "
            "tools/rust_doc_ratchet.py."
        ),
        "provenance": dict(sorted(provenance.items())),
        "schema_version": RUST_DOC_CEILING_SCHEMA_VERSION,
        "undocumented": undocumented,
        "undocumented_files": files,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def measure(root: Path, manifest: str) -> tuple[int, int, str]:
    """Run the lint and return the count, the file count and the rustc version.

    Raises
    ------
    RatchetError
        Cargo is not installed, so nothing was measured. A check that cannot
        run must say so rather than report zero.
    """
    if shutil.which("cargo") is None:
        raise RatchetError("cargo is not installed, so no measurement was taken")
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["cargo", "rustc", "--lib", "--manifest-path", manifest, "--", "-W", "missing_docs"],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=3600,
        check=False,
    )
    undocumented, files = read_rustc_stderr(completed.stderr)
    version = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["rustc", "--version"],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=120,
        check=False,
    ).stdout.strip()
    return undocumented, files, version


def main(argv: list[str] | None = None) -> int:
    """Compare the crate against its ceiling, or lower the ceiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", default="engine/Cargo.toml")
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    parser.add_argument(
        "--update",
        action="store_true",
        help="lower the ceiling to the measured figure; it can never be raised",
    )
    args = parser.parse_args(argv)
    try:
        undocumented, files, version = measure(args.repo.resolve(), args.manifest)
    except RatchetError as error:
        print(f"error: {error}")
        return 2
    if args.update:
        sha = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=args.repo,
            timeout=120,
            check=False,
        ).stdout.strip()
        try:
            write_ceiling(
                args.ceiling,
                undocumented=undocumented,
                files=files,
                provenance={
                    "argv": f"cargo rustc --lib --manifest-path {args.manifest} -- -W missing_docs",
                    "rustc": version,
                    "source_sha256": sha,
                },
            )
        except RatchetError as error:
            print(f"error: {error}")
            return 2
        print(f"Ceiling now {undocumented} undocumented items in {files} files.")
        return 0
    try:
        ceiling = read_ceiling(args.ceiling)
    except (RatchetError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2
    verdict = compare(undocumented, ceiling)
    print(verdict.summary())
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
