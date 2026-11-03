# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete existing MAT ABI consumer for native Go coverage

"""Run the unchanged whole ABI module and conserve its actual collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.go_coverage_images import register
from tools.go_coverage_consumer_inputs import (
    ABIExecution,
    MODULE as MODULE,
    Membership,
    validate_collect_only_result,
    validate_complete_execution,
    validate_source_projection,
)


def save(path: Path, value: object) -> None:
    """Create one result without overwriting a prior observation or native output."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def execute(job: Path, repository: Path, *, collect_only: bool = False) -> int:
    """Execute every original case in the gate's isolated library-path projection.

    Parameters
    ----------
    job : pathlib.Path
        Fresh gate allocation with closed environment and image declarations.
    repository : pathlib.Path
        Current exact source supplying the unchanged test and Python models.
    collect_only : bool, optional
        Capture the actual full collection before executing that same cohort.

    Returns
    -------
    int
        Zero only for a complete successful collection or execution.

    Raises
    ------
    RuntimeError
        Test source, origin, membership or actual execution differs.
    """
    if sys.flags.optimize:
        raise RuntimeError("Native ABI qualification requires unoptimized assertions")
    register(job, "tests")
    subject = job / "subject"
    original = repository / MODULE
    copied = subject / MODULE
    expected = validate_source_projection(original, copied)
    sys.path[:0] = [str(subject), str(repository / "src")]
    import pytest

    observer = Membership(job, copied, collect_only)
    arguments = [
        "-q",
        "-o",
        "addopts=",
        "-o",
        "pythonpath=",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(subject),
        "--confcutdir",
        str(subject / "tests"),
        str(copied),
    ]
    if collect_only:
        arguments.append("--collect-only")
    else:
        arguments += ["--basetemp", str(job / "pytest-tmp"), "--junitxml", str(job / "abi.xml")]
    status = pytest.main(arguments, plugins=[observer])
    unchanged = expected == hashlib.sha256(original.read_bytes()).hexdigest()
    unchanged = unchanged and expected == hashlib.sha256(copied.read_bytes()).hexdigest()
    if collect_only:
        validate_collect_only_result(status, unchanged, observer.nodes)
        save(job / "collection.json", observer.nodes)
        save(
            job / "collection-process.json",
            {
                "pid": os.getpid(),
                "test_sha256": expected,
                "pytest_exit": int(status),
                "source_unchanged": unchanged,
                "module_origin": observer.origin,
            },
        )
        return 0
    cases = ET.parse(job / "abi.xml").findall(".//testcase")
    rejected = [
        case.attrib
        for case in cases
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
    ]
    result: ABIExecution = {
        "pid": os.getpid(),
        "pytest_exit": int(status),
        "junit_cases": len(cases),
        "rejected": rejected,
        "actual_collection": observer.nodes,
        "passed": observer.passed,
        "other_reports": observer.other,
        "test_sha256": expected,
        "source_unchanged": unchanged,
        "measurement": "630 existing ABI cases, not630 new tests",
    }
    save(job / "execution.json", result)
    validate_complete_execution(result)
    return 0


def main() -> int:
    """Enter only with an explicit gate allocation and original repository source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    return execute(
        args.job.resolve(strict=True),
        args.repository.resolve(strict=True),
        collect_only=args.collect_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
