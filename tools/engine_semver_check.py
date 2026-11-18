#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — engine crate public API compatibility gate

"""Refuse an incompatible public change of the engine crate outside a major release.

``cargo semver-checks`` compares the engine crate of this tree with a release published on
crates.io. The release gate of 4.2.0 compared only public functions and the shapes of public
structs and enums, with two scripts that miss traits, impls, attributes and re-exports.

The baseline follows the version in ``engine/Cargo.toml``:

- **The version is already published.** Main is between releases. That release is the baseline,
  and the change is classed as a minor release. An addition passes and a breaking change is
  refused, because the next release made from this tree is not a major one unless its version
  says so.
- **The version is not published yet.** This is the release bump. The tool's own rule applies:
  the highest published version below it is the baseline, and the release type follows from the
  two version numbers. A major bump admits a breaking change; a minor or patch bump does not.

Only the ``gpu`` feature is enabled, on both sides. The default ``z3`` feature is left out,
because its bundled Z3 C++ build would dominate the job. The ``supervisor`` module behind it is
therefore not compared, and every run says so.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

RunCommand = Callable[..., subprocess.CompletedProcess[str]]
FetchIndex = Callable[[str], str]

ENGINE_MANIFEST = Path("engine/Cargo.toml")
CHECKED_FEATURES = ("gpu",)
UNCHECKED_FEATURE_NOTE = (
    "Not compared: the items behind the default feature z3 (the supervisor module), whose bundled "
    "Z3 C++ build this check leaves out."
)
SPARSE_INDEX = "https://index.crates.io"
CRATE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
INDEX_TIMEOUT_SECONDS = 60.0
CHECK_TIMEOUT_SECONDS = 3600
LINT_FAILURE_EXIT_CODE = 100


def crate_identity(manifest: Path) -> tuple[str, str]:
    """Return the package name and version that a Cargo manifest declares.

    Parameters
    ----------
    manifest : Path
        A ``Cargo.toml`` with a ``[package]`` table.

    Returns
    -------
    tuple of str
        The package name and its version.

    Raises
    ------
    ValueError
        If the manifest has no package name or version.
    """
    package = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {})
    name, version = package.get("name"), package.get("version")
    if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
        raise ValueError(f"{manifest} declares no package name and version.")
    return name, version


def sparse_index_path(name: str) -> str:
    """Return the path of a crate's file in the crates.io sparse index.

    Parameters
    ----------
    name : str
        Crate name.

    Returns
    -------
    str
        The index path, lower case, as Cargo's index layout defines it.
    """
    lower = name.lower()
    if len(lower) == 1:
        return f"1/{lower}"
    if len(lower) == 2:
        return f"2/{lower}"
    if len(lower) == 3:
        return f"3/{lower[0]}/{lower}"
    return f"{lower[:2]}/{lower[2:4]}/{lower}"


def fetch_sparse_index(name: str) -> str:
    """Download a crate's sparse-index file from crates.io.

    Parameters
    ----------
    name : str
        Crate name.

    Returns
    -------
    str
        The file's text: one JSON object per published version.

    Raises
    ------
    ValueError
        If ``name`` is not a crate name, or the index URL is not HTTPS on the crates.io index
        host. Either check runs before anything is opened, so no other scheme (``file:`` or a
        custom one) and no other path can be read.
    """
    if not CRATE_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a crate name.")
    url = f"{SPARSE_INDEX}/{sparse_index_path(name)}"
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "index.crates.io":
        raise ValueError(f"Refusing to open {url!r}: only https://index.crates.io is read.")
    request = urllib.request.Request(
        url, headers={"User-Agent": "sc-neurocore-engine-semver-check"}
    )
    # Bandit B310 is answered by the two checks above: only https://index.crates.io is opened.
    with urllib.request.urlopen(request, timeout=INDEX_TIMEOUT_SECONDS) as response:  # nosec B310
        body: bytes = response.read()
    return body.decode("utf-8")


def published_versions(index_text: str) -> set[str]:
    """Return the versions in a sparse-index file that are published and not yanked.

    Parameters
    ----------
    index_text : str
        A crate's sparse-index file.

    Returns
    -------
    set of str
        The versions that a baseline may name.

    Raises
    ------
    ValueError
        If a line is not a version record, since a misread index must not decide the baseline.
    """
    versions: set[str] = set()
    for number, line in enumerate(index_text.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("vers"), str):
            raise ValueError(f"Sparse-index line {number} is not a version record.")
        if record.get("yanked") is not True:
            versions.add(record["vers"])
    return versions


def check_command(manifest: Path, version: str, published: set[str]) -> list[str]:
    """Return the ``cargo semver-checks`` command for a crate version and its published versions.

    Parameters
    ----------
    manifest : Path
        The crate's manifest, relative to the repository root.
    version : str
        The version the manifest declares.
    published : set of str
        The crate's published, unyanked versions.

    Returns
    -------
    list of str
        The command. When ``version`` is published, it names that release as the baseline and
        classes the change as minor. Otherwise it leaves both to the tool's rule.
    """
    command = [
        "cargo",
        "semver-checks",
        "check-release",
        "--manifest-path",
        manifest.as_posix(),
        "--only-explicit-features",
        "--features",
        ",".join(CHECKED_FEATURES),
    ]
    if version in published:
        command += ["--baseline-version", version, "--release-type", "minor"]
    return command


def main(
    argv: Sequence[str] | None = None,
    *,
    run: RunCommand = subprocess.run,
    fetch: FetchIndex = fetch_sparse_index,
) -> int:
    """Compare the engine crate with its published baseline; return 0 only for a compatible change.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; the process arguments when omitted.
    run : callable, optional
        Runs one command, with the signature of :func:`subprocess.run`.
    fetch : callable, optional
        Returns a crate's sparse-index file.

    Returns
    -------
    int
        0 when the check completed and found no incompatible change. 1 when it found one, or when
        it could not complete: a check that did not run never passes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    try:
        name, version = crate_identity(root / ENGINE_MANIFEST)
        published = published_versions(fetch(name))
    except (OSError, ValueError) as error:
        print(f"The baseline could not be determined: {error}", file=sys.stderr)
        return 1
    command = check_command(ENGINE_MANIFEST, version, published)
    if version in published:
        print(f"{name} {version} is published: it is the baseline; the change is classed as minor.")
    else:
        print(f"{name} {version} is not published: the highest published version below it is used.")
    print(UNCHECKED_FEATURE_NOTE)
    print("+ " + " ".join(command), flush=True)
    try:
        result = run(command, cwd=root, check=False, timeout=CHECK_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"cargo semver-checks did not run: {error}", file=sys.stderr)
        return 1
    if result.returncode == 0:
        print(f"No incompatible public change of {name} against its baseline.")
        return 0
    if result.returncode == LINT_FAILURE_EXIT_CODE:
        print(
            f"{name} has an incompatible public change. "
            "Revert it, or release it as a new major version.",
            file=sys.stderr,
        )
    else:
        print(f"cargo semver-checks did not complete (exit {result.returncode}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
