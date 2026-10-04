#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go documentation debt may fall, never rise

"""Protect original Go documentation debt with actual native declaration identities.

Native Git discovers tracked and nonignored untracked Go source. Go's parser
measures those exact bytes and emits package, file and receiver identities.
An independently parsed immutable Git baseline protects the original cohort,
individual unresolved declarations and scalar ceiling. Documenting one case
cannot authorize a new undocumented declaration with the same aggregate count.

``--update`` may lower the scalar and individual allowances. It records the
current native source manifest and original source evidence. Explicitly ignored
and external source ownership requires separate qualification.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from tools.doc_debt_ceiling import RatchetError, Verdict, read_ceiling
from tools.doc_debt_ceiling import compare as _compare
from tools.doc_debt_ceiling import write_ceiling as _write_ceiling
from tools.docstring_policy_git import git_bytes
from tools.docstring_policy_scope import PolicyError
from tools.go_doc_measurement import (
    GO_COVERAGE_TOOL,
    GoDocMeasurement,
    GoMeasurementError,
    qualify_repository,
)
from tools.go_doc_findings import measure_findings
from tools.go_doc_history import INDIVIDUAL_CEILING_SCHEMA, protect_debt

#: Contract version of the ceiling record.
GO_DOC_CEILING_SCHEMA_VERSION = "sc-neurocore.go-doc-ceiling.v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Where the committed ceiling lives, beside the tool that maintains it.
#:
#: Not beside the code it constrains, because its scope is Git-discovered
#: ``.go`` file in the repository and has no single directory to sit in.
DEFAULT_CEILING = REPO_ROOT / "tools" / "go_doc_ceiling.json"
#: What the record means, written for whoever opens the file before the tool.
CEILING_NOTE = (
    "Undocumented exported declarations in tracked and nonignored untracked "
    ".go files, measured with Go's parser through tools/godoc_coverage. Original "
    "individual cases are protected. This ceiling may fall and may not rise: "
    "see tools/go_doc_ratchet.py."
)


def compare(measured: int, ceiling: int) -> Verdict:
    """Return the verdict for one measurement against the ceiling."""
    return _compare(measured, ceiling, language="Go")


def write_ceiling(
    path: Path, *, undocumented: int, files: int, provenance: Mapping[str, object]
) -> None:
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
        schema_version=INDIVIDUAL_CEILING_SCHEMA
        if "debt_cases" in provenance
        else GO_DOC_CEILING_SCHEMA_VERSION,
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
    qualified = _measurement(root)
    return qualified.undocumented, qualified.files, qualified.version


def _measurement(root: Path) -> GoDocMeasurement:
    try:
        return qualify_repository(root)
    except GoMeasurementError as error:
        raise RatchetError(f"the Go coverage tool did not produce a figure: {error}") from error


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
        qualified = _measurement(root)
        individual = measure_findings(root, qualified.source.paths)
        if individual.measurement.summary != qualified.summary:
            raise GoMeasurementError(
                "The current Go summary changed during individual measurement."
            )
        protected = protect_debt(root, individual, args.ceiling, allow_create=args.update)
        qualified.verify(root, tracked=True)
        undocumented, files, version = qualified.undocumented, qualified.files, qualified.version
    except (RatchetError, GoMeasurementError) as error:
        print(f"error: {error}")
        return 2
    if args.update:
        try:
            sha = git_bytes(root, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
            qualified.verify(root, tracked=True)
            write_ceiling(
                args.ceiling,
                undocumented=undocumented,
                files=files,
                provenance={
                    "argv": ["go", "run", GO_COVERAGE_TOOL],
                    "scope_argv": [
                        "git",
                        "ls-files",
                        "--cached",
                        "--others",
                        "--exclude-standard",
                        "-z",
                        "--",
                        "*.go",
                    ],
                    "go": version,
                    "source_sha256": sha,
                    "git_revision": sha,
                    "measurement": qualified.to_public_dict(),
                    "individual_measurement": individual.to_public_dict(),
                    "original_baseline": protected.to_public_dict(),
                    "debt_cases": [case.to_public_dict() for case in sorted(individual.cases)],
                },
            )
        except (RatchetError, GoMeasurementError, PolicyError, UnicodeError) as error:
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
