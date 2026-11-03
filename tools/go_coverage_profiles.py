# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Go profile admission and complete MAT statement checks

"""Check complete native profiles without filtering or changing their block universe.

The caller produces both files with the real Go tools and retains their source,
image and process custody. Parsing text does not authenticate its producer.
Go statement counters do not provide general branch or MC/DC instrumentation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import TypeAlias

MODULE = "github.com/anulum/sc-neurocore/accel/"
TARGET_STATEMENTS = {
    MODULE + "services/mat.go": 50,
    MODULE + "services/sc_resetting_mat.go": 45,
    MODULE + "mat/mat.go": 26,
    MODULE + "sc_resetting_mat/sc_resetting_mat.go": 23,
}
DIAGNOSTIC_SOURCES = (
    MODULE + "mat/coverage_snapshot.go",
    MODULE + "sc_resetting_mat/coverage_snapshot.go",
    MODULE + "internal/coverageprofile/snapshot.go",
)
_ROW = re.compile(r"(.+):(\d+)\.(\d+),(\d+)\.(\d+) (\d+) (\d+)")
Coordinates: TypeAlias = tuple[str, int, int, int, int]


@dataclass(frozen=True)
class Block:
    """One native counter with its exact source geometry and statement count.

    Attributes
    ----------
    coordinates : tuple
        Raw filename and starting/ending one-based line and column.
    statements : int
        The Go compiler's statement denominator for this block.
    count : int
        Observed native counter, without interpreting it as unique calls.
    """

    coordinates: Coordinates
    statements: int
    count: int


def read_profile(path: Path) -> dict[Coordinates, Block]:
    """Read every block in a native atomic text profile.

    Parameters
    ----------
    path : pathlib.Path
        Complete output from ``go tool covdata textfmt``.

    Returns
    -------
    dict
        All blocks keyed by their original filename and coordinates.

    Raises
    ------
    ValueError
        The mode, row syntax, coordinates or unique block identities are invalid.
    OSError
        The complete native file cannot be read.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0] != "mode: atomic":
        raise ValueError("A nonempty native atomic profile is required")
    result: dict[Coordinates, Block] = {}
    for number, line in enumerate(lines[1:], start=2):
        match = _ROW.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed native profile row {number}")
        filename = match[1]
        start_line, start_column, end_line, end_column, statements, count = (
            int(match[index]) for index in range(2, 8)
        )
        if (
            "\x00" in filename
            or "\\" in filename
            or ".." in PurePosixPath(filename).parts
            or min(start_line, start_column, end_line, end_column) < 1
            or (end_line, end_column) < (start_line, start_column)
        ):
            raise ValueError(f"Invalid native source geometry at row {number}")
        key = (filename, start_line, start_column, end_line, end_column)
        if key in result:
            raise ValueError(f"Duplicate native block at row {number}")
        result[key] = Block(key, statements, count)
    return result


def compare_profiles(before: Path, after: Path) -> dict[str, object]:
    """Require every maintained MAT statement in the conserved native universe.

    Parameters
    ----------
    before : pathlib.Path
        Real before-work profile of the same instrumented source cohort.
    after : pathlib.Path
        Real combined native-test and complete ABI-consumer profile.

    Returns
    -------
    dict
        Per-source statement results and the complete native block count.

    Raises
    ------
    ValueError
        Native blocks or denominators change, a subject is absent or shrunk,
        the before-work subject was already executed, or a statement is missing.
    """
    initial = read_profile(before)
    final = read_profile(after)
    if initial.keys() != final.keys():
        raise ValueError("The complete native block universe changed")
    if any(block.statements != final[key].statements for key, block in initial.items()):
        raise ValueError("A native statement denominator changed")
    sources: list[dict[str, object]] = []
    for filename, minimum in TARGET_STATEMENTS.items():
        selected = [block for block in final.values() if block.coordinates[0] == filename]
        total = sum(block.statements for block in selected)
        if total < minimum:
            raise ValueError(f"Native subject absent or statement scope shrunk: {filename}")
        if any(initial[block.coordinates].count for block in selected if block.statements):
            raise ValueError(f"Before-work subject was already executed: {filename}")
        missing = [block.coordinates for block in selected if block.statements and block.count == 0]
        if missing:
            raise ValueError(f"Uncovered native statements in {filename}: {missing}")
        sources.append(
            {
                "path": filename,
                "covered_statements": total,
                "total_statements": total,
                "blocks": len(selected),
                "zero_statement_blocks": sum(block.statements == 0 for block in selected),
            }
        )
    return {
        "passed": True,
        "measurement": "native Go statements",
        "sources": sources,
        "native_blocks_retained": len(final),
        "target_blocks": sum(block.coordinates[0] in TARGET_STATEMENTS for block in final.values()),
        "branch_or_mcdc_qualification": False,
        "all_function_qualification": False,
    }


def profile_observations(path: Path) -> dict[str, object]:
    """Describe all four subjects even when missing counters must refuse promotion.

    Parameters
    ----------
    path : pathlib.Path
        Complete native text profile, without source or block filtering.

    Returns
    -------
    dict
        Actual per-subject covered and total statements, including absent sources.
    """
    blocks = read_profile(path)
    rows = []
    for filename in TARGET_STATEMENTS:
        selected = [block for block in blocks.values() if block.coordinates[0] == filename]
        rows.append(
            {
                "path": filename,
                "present": bool(selected),
                "covered_statements": sum(block.statements for block in selected if block.count),
                "total_statements": sum(block.statements for block in selected),
                "blocks": len(selected),
            }
        )
    return {
        "native_blocks": len(blocks),
        "sources": rows,
        "measurement": "native Go statement observations, not a promotion verdict",
    }


def diagnostic_statements(path: Path) -> list[dict[str, object]]:
    """Require every statement in each newly introduced native snapshot source.

    Parameters
    ----------
    path : pathlib.Path
        Actual complete profile produced by the current tagged native images.

    Returns
    -------
    list of dict
        Complete per-file diagnostic statement totals, separate from model scope.

    Raises
    ------
    ValueError
        A diagnostic source is absent, empty or retains any uncovered statement.
    """
    blocks = read_profile(path)
    result = []
    for filename in DIAGNOSTIC_SOURCES:
        selected = [block for block in blocks.values() if block.coordinates[0] == filename]
        total = sum(block.statements for block in selected)
        if total == 0 or any(block.statements and not block.count for block in selected):
            raise ValueError(f"Missing or uncovered native diagnostic statements: {filename}")
        result.append({"path": filename, "covered_statements": total, "total_statements": total})
    return result


def main() -> int:
    """Write the actual complete observation and fail on any refused native comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_profiles(args.before, args.after)
    except (OSError, ValueError) as error:
        result = {"passed": False, "error": repr(error)}
    try:
        result["observed_after"] = profile_observations(args.after)
    except (OSError, ValueError) as error:
        result["observed_after"] = None
        result["observation_error"] = repr(error)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except OSError as error:
        print("Native profile result could not be created: " + repr(error), file=sys.stderr)
        return 1
    return 0 if result["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
