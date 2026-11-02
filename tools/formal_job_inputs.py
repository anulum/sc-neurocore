# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SymbiYosys invocation and source identity

"""Resolve existing formal-job directory conventions using SymbiYosys itself."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class FormalJobInputs:
    """Retain the admitted job, working directory and complete copied sources.

    Parameters
    ----------
    repository : Path
        Resolved repository root containing every admitted input.
    job : Path
        Resolved SymbiYosys configuration file.
    directory : Path
        Directory from which its existing source references resolve.
    sources : tuple of Path
        Unique configuration and copied source files, in sorted order.
    """

    repository: Path
    job: Path
    directory: Path
    sources: tuple[Path, ...]

    def source_hashes(self) -> dict[str, str]:
        """Hash the actual configuration and every copied source file.

        Returns
        -------
        dict of str to str
            Repository-relative paths and their lowercase SHA-256 digests.
        """
        result = {}
        for source in self.sources:
            with source.open("rb") as stream:
                result[source.relative_to(self.repository).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        return result


def resolve_job(repository: Path, job: Path) -> FormalJobInputs:
    """Select the existing directory convention without rewriting a formal job.

    Parameters
    ----------
    repository : Path
        Checkout containing the job and all copied sources.
    job : Path
        Absolute or repository-relative SymbiYosys configuration path.

    Returns
    -------
    FormalJobInputs
        Invocation directory and exact source closure reported by ``sby``.

    Raises
    ------
    ValueError
        If the job is missing, escapes the checkout, has no external sources,
        or its sources are missing, outside the checkout or ambiguous.
    OSError
        If the checkout or SymbiYosys executable is unavailable.
    subprocess.SubprocessError
        If SymbiYosys cannot parse the actual job within thirty seconds.
    """
    repository = repository.resolve(strict=True)
    job = (repository / job).resolve()
    if not job.is_relative_to(repository) or job.suffix != ".sby" or not job.is_file():
        raise ValueError("The formal job must be an existing .sby file inside the checkout.")
    tasks = subprocess.run(
        ["sby", "--dumptasks", str(job)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.splitlines()
    parsed = subprocess.run(
        ["sby", "--dumpfiles", str(job), *tasks],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    files = parsed.stdout.strip().splitlines()
    if not files:
        raise ValueError("The formal job has no complete copied-source inventory.")
    candidates: list[tuple[Path, tuple[Path, ...]]] = []
    for directory in dict.fromkeys((job.parent, repository)):
        sources = tuple(sorted({(directory / name).resolve() for name in files}))
        if all(source.is_file() and source.is_relative_to(repository) for source in sources):
            candidates.append((directory, sources))
    if not candidates:
        raise ValueError(
            "Formal sources are missing or outside the checkout under both conventions."
        )
    if len({sources for _directory, sources in candidates}) != 1:
        raise ValueError(
            "Formal sources are ambiguous between job-relative and checkout-relative paths."
        )
    directory, sources = candidates[0]
    return FormalJobInputs(repository, job, directory, tuple(sorted({job, *sources})))


def main(argv: Sequence[str] | None = None) -> int:
    """Print safe invocation paths or their source-bound JSON representation.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; process arguments are used when omitted.

    Returns
    -------
    int
        Zero for admitted inputs, two for missing, invalid or unavailable inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        inputs = resolve_job(args.repository, args.job)
        directory = inputs.directory.relative_to(inputs.repository).as_posix()
        argument = inputs.job.relative_to(inputs.directory).as_posix()
        if args.json:
            print(
                json.dumps(
                    {
                        "directory": directory,
                        "argument": argument,
                        "sources": inputs.source_hashes(),
                    }
                )
            )
        else:
            print(directory)
            print(argument)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
