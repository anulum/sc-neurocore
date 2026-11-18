#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust crate formatting gate

"""Check that every Rust crate under ``crates/`` is formatted as ``cargo fmt`` formats it.

The engine and the documentation symbol producer have formatting gates of their own. The
crates under ``crates/`` had none, so formatting drift went unseen. One command checks every
crate the directory holds, so a crate added later is checked without a new gate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

RunCommand = Callable[..., subprocess.CompletedProcess[str]]
CRATES_DIRECTORY = "crates"


def crate_manifests(root: Path) -> list[Path]:
    """Return the manifest of every crate under ``crates/``, relative to ``root``, in name order.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    list of Path
        One ``crates/<name>/Cargo.toml`` per crate.

    Raises
    ------
    ValueError
        If no crate manifest exists, since a check of no crate would pass without checking.
    """
    manifests = sorted(
        path.relative_to(root) for path in (root / CRATES_DIRECTORY).glob("*/Cargo.toml")
    )
    if not manifests:
        raise ValueError(f"No crate manifest under {CRATES_DIRECTORY}/: nothing to check.")
    return manifests


def unformatted_crates(root: Path, run: RunCommand = subprocess.run) -> tuple[int, list[str]]:
    """Run ``cargo fmt --check`` on each crate and name the crates whose formatting differs.

    Parameters
    ----------
    root : Path
        Repository root.
    run : callable, optional
        Runs one command, with the signature of :func:`subprocess.run`.

    Returns
    -------
    tuple
        The number of crates checked and the manifests of those that differ. Each difference is
        written to standard output as ``cargo fmt`` reports it.
    """
    manifests = crate_manifests(root)
    differing: list[str] = []
    for manifest in manifests:
        result = run(
            ["cargo", "fmt", "--check", "--manifest-path", manifest.as_posix()],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if result.returncode != 0:
            differing.append(manifest.as_posix())
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
    return len(manifests), differing


def main(argv: Sequence[str] | None = None, *, run: RunCommand = subprocess.run) -> int:
    """Check every crate and return 0 when all are formatted, 1 otherwise.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; the process arguments when omitted.
    run : callable, optional
        Runs one command, with the signature of :func:`subprocess.run`.

    Returns
    -------
    int
        0 when every crate is formatted, 1 when one is not or none exists.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    try:
        checked, differing = unformatted_crates(root, run)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if differing:
        print(
            f"{len(differing)} of {checked} crates are not formatted: {', '.join(differing)}. "
            "Run cargo fmt --manifest-path on each.",
            file=sys.stderr,
        )
        return 1
    print(f"All {checked} crates under {CRATES_DIRECTORY}/ are formatted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
