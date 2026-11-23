# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine Python-binding location rule

"""Refuse PyO3 functions that sit outside the engine's binding modules.

Every ``#[pyfunction]`` in a tracked Rust file under ``engine/src`` belongs to a
binding module: a file under a ``bindings`` directory, or a file whose stem is
``bindings``. Model, integrator and facade files hold no Python surface, so the
binding layer of each domain stays in one recognisable place. The rule covers
every tracked engine file, including new ones, and counts nothing.

The bare attribute ``#[pyfunction]``, its argument form ``#[pyfunction(...)]``
and the qualified form ``#[pyo3::pyfunction]`` are all recognised when they open
a line; a mention inside a comment is not an attribute.

Usage::

    python tools/engine_bindings_location.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ENGINE_SOURCE = "engine/src"
BINDINGS = "bindings"

#: An attribute line opening ``#[pyfunction]``, ``#[pyfunction(...)]`` or
#: ``#[pyo3::pyfunction]``, with the optional spaces Rust allows between tokens.
PYFUNCTION_ATTRIBUTE = re.compile(r"^\s*#\s*\[\s*(?:pyo3\s*::\s*)?pyfunction\b")

_GIT_TIMEOUT_SECONDS = 60


class BindingsLocationError(RuntimeError):
    """Raised when the tracked engine sources cannot be listed or read."""


@dataclass(frozen=True)
class MisplacedBinding:
    """One ``#[pyfunction]`` attribute found outside a binding module.

    Parameters
    ----------
    path : str
        Repository-relative POSIX path of the Rust file.
    line : int
        One-based line number of the attribute.
    """

    path: str
    line: int


def is_binding_module(path: str) -> bool:
    """Return whether a repository path names an engine binding module.

    Parameters
    ----------
    path : str
        Repository-relative POSIX path of a Rust file.

    Returns
    -------
    bool
        True when a directory component is ``bindings`` or the file stem is.
    """
    pure = PurePosixPath(path)
    return BINDINGS in pure.parts[:-1] or pure.stem == BINDINGS


def tracked_engine_sources(repo: Path) -> list[str]:
    """List the tracked Rust files under ``engine/src``.

    Parameters
    ----------
    repo : Path
        Repository working directory.

    Returns
    -------
    list of str
        Sorted repository-relative POSIX paths ending in ``.rs``.

    Raises
    ------
    BindingsLocationError
        If Git cannot run, times out or reports failure.
    """
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        env.pop(name, None)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z", "--", ENGINE_SOURCE],
            capture_output=True,
            check=False,
            env=env,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BindingsLocationError(f"git ls-files could not run: {exc}") from exc
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise BindingsLocationError(f"git ls-files failed: {message}")
    paths = result.stdout.decode("utf-8").split("\0")
    return sorted(path for path in paths if path.endswith(".rs"))


def misplaced_bindings(repo: Path, paths: Sequence[str]) -> list[MisplacedBinding]:
    """Return every ``#[pyfunction]`` attribute outside a binding module.

    Parameters
    ----------
    repo : Path
        Repository working directory.
    paths : sequence of str
        Repository-relative Rust paths to inspect.

    Returns
    -------
    list of MisplacedBinding
        One entry per attribute, in path and line order.

    Raises
    ------
    BindingsLocationError
        If a listed file cannot be read as UTF-8 text.
    """
    found: list[MisplacedBinding] = []
    for path in paths:
        if is_binding_module(path):
            continue
        try:
            text = (repo / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise BindingsLocationError(f"{path} could not be read: {exc}") from exc
        for number, line in enumerate(text.splitlines(), start=1):
            if PYFUNCTION_ATTRIBUTE.match(line):
                found.append(MisplacedBinding(path, number))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """Check the binding-location rule and report each misplaced attribute.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; the process arguments are used when omitted.

    Returns
    -------
    int
        Zero when every PyO3 function sits in a binding module, one for a
        misplaced attribute, and two when the sources cannot be listed or read.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check", action="store_true", required=True, help="refuse misplaced bindings"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo = args.repo.resolve()
    try:
        paths = tracked_engine_sources(repo)
        found = misplaced_bindings(repo, paths)
    except BindingsLocationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if found:
        print("PyO3 functions outside an engine binding module:", file=sys.stderr)
        for entry in found:
            print(f"  {entry.path}:{entry.line}", file=sys.stderr)
        print(
            "Move each #[pyfunction] into its domain's binding module: a file under a "
            "'bindings' directory or a file named bindings.rs.",
            file=sys.stderr,
        )
        return 1
    print(f"Engine bindings location accepted: {len(paths)} tracked Rust files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
