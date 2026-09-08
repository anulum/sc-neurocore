#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go documentation debt may fall, never rise

"""A ratchet on the undocumented exported declarations in this repository's Go.

Go was recorded as *not measured* here, for the stated reason that no Go
documentation linter is installed. That reason was true of the linters and
false of the language: ``go/parser`` is the front end those linters are built
on, it ships with the toolchain, and the toolchain is present. The measurement
is therefore taken by ``tools/godoc_coverage``, which parses the same way the
compiler does; the first figure it produced was 1532 undocumented declarations
across 420 files.

The scope is **every tracked ``.go`` file**, taken from ``git ls-files`` rather
than from a list of directories or a walk of the tree. Both alternatives were
tried and both were wrong in a way that matters: a list is silently blind to
any file nobody adds to it, and a filesystem walk of the Go tree found 18 057
files because a virtual environment lives inside it, so the figure would have
described a Go toolchain rather than this project.

Debt may fall and the ceiling follows it down; debt may not rise, and this tool
cannot raise the ceiling at all. The rule itself is shared with every other
language in ``tools/doc_debt_ceiling.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tools.doc_debt_ceiling import RatchetError, Verdict, read_ceiling
from tools.doc_debt_ceiling import compare as _compare
from tools.doc_debt_ceiling import write_ceiling as _write_ceiling
from tools.documentation_debt import GO_COVERAGE_TOOL, go_scope, measure_go_coverage

#: Contract version of the ceiling record.
GO_DOC_CEILING_SCHEMA_VERSION = "sc-neurocore.go-doc-ceiling.v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Where the committed ceiling lives, beside the tool that maintains it.
#:
#: Not beside the code it constrains, because that scope is every tracked
#: ``.go`` file in the repository and has no single directory to sit in.
DEFAULT_CEILING = REPO_ROOT / "tools" / "go_doc_ceiling.json"
#: What the record means, written for whoever opens the file before the tool.
CEILING_NOTE = (
    "Undocumented exported declarations in every tracked .go file, measured "
    "with Go's own parser through tools/godoc_coverage. This ceiling may fall "
    "and may not rise: see tools/go_doc_ratchet.py."
)


def compare(measured: int, ceiling: int) -> Verdict:
    """Return the verdict for one measurement against the ceiling."""
    return _compare(measured, ceiling, language="Go")


def write_ceiling(path: Path, *, undocumented: int, files: int, provenance: dict[str, str]) -> None:
    """Write the Go ceiling record, refusing to raise an existing one.

    Raises
    ------
    RatchetError
        The new figure is above the recorded one. Lowering is the tool's job;
        raising is a decision, and it belongs in a diff somebody signed.
    """
    _write_ceiling(
        path,
        undocumented=undocumented,
        files=files,
        note=CEILING_NOTE,
        schema_version=GO_DOC_CEILING_SCHEMA_VERSION,
        provenance=provenance,
    )


def measure(root: Path) -> tuple[int, int, str]:
    """Run the coverage tool and return the count, the file count and the version.

    Raises
    ------
    RatchetError
        Go is not installed, or the tool could not read its scope. A check that
        cannot run must say so rather than report zero: an unreadable scope and
        a documented one are the same number and opposite facts.
    """
    if shutil.which("go") is None:
        raise RatchetError("go is not installed, so no measurement was taken")
    paths = go_scope(root)
    if not paths:
        raise RatchetError("no tracked .go files were found, so nothing was measured")
    try:
        summary = measure_go_coverage(root, paths)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RatchetError(f"the Go coverage tool did not produce a figure: {error}") from error
    version = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["go", "version"],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=120,
        check=False,
    ).stdout.strip()
    return summary["undocumented"], summary["files_with_findings"], version


def main(argv: list[str] | None = None) -> int:
    """Compare the Go surface against its ceiling, or lower the ceiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    parser.add_argument(
        "--update",
        action="store_true",
        help="lower the ceiling to the measured figure; it can never be raised",
    )
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    try:
        undocumented, files, version = measure(root)
    except RatchetError as error:
        print(f"error: {error}")
        return 2
    if args.update:
        sha = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=120,
            check=False,
        ).stdout.strip()
        try:
            write_ceiling(
                args.ceiling,
                undocumented=undocumented,
                files=files,
                provenance={
                    "argv": f"git ls-files -- '*.go' | go run {GO_COVERAGE_TOOL}",
                    "go": version,
                    "source_sha256": sha,
                },
            )
        except RatchetError as error:
            print(f"error: {error}")
            return 2
        print(f"Ceiling now {undocumented} undocumented declarations in {files} files.")
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
