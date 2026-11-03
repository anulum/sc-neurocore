# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete original ABI collection and execution custody

"""Observe real Pytest hooks and conserve complete unchanged ABI outcomes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Protocol, TypedDict

MODULE = "tests/test_mat_family_c_abi.py"


class CollectedItem(Protocol):
    """Describe the node identity read from a genuine Pytest collection item."""

    nodeid: str


class CollectedSession(Protocol):
    """Describe the actual Pytest session collection without importing the framework early."""

    items: Sequence[CollectedItem]


class CompletedReport(Protocol):
    """Describe fields read from genuine framework reports, never fabricated report objects."""

    nodeid: str
    when: str
    outcome: str
    passed: bool
    failed: bool
    skipped: bool


class ABIExecution(TypedDict):
    """Keep the original complete consumer result and the actual framework observations."""

    pid: int
    pytest_exit: int
    junit_cases: int
    rejected: list[dict[str, str]]
    actual_collection: list[str]
    passed: list[str]
    other_reports: list[dict[str, str]]
    test_sha256: str
    source_unchanged: bool
    measurement: str


def validate_source_projection(original: Path, copied: Path) -> str:
    """Read both real test files and require the exact unchanged whole ABI source bytes."""
    expected = hashlib.sha256(original.read_bytes()).hexdigest()
    if hashlib.sha256(copied.read_bytes()).hexdigest() != expected:
        raise RuntimeError("The existing ABI module changed in the coverage projection")
    return expected


def validate_node_count(nodes: list[str]) -> None:
    """Require all630 distinct original ABI cases without selecting or fabricating members."""
    if len(nodes) != 630 or len(set(nodes)) != 630:
        raise RuntimeError("Complete original ABI membership must contain all630 cases")


def validate_collection_origin(origin: str | None, copied: Path) -> None:
    """Bind the actual imported module file to the real source projection used by this consumer."""
    if origin is None or Path(origin).resolve() != copied.resolve():
        raise RuntimeError("The ABI module was imported from another source")


def validate_collection_order(nodes: list[str], declared: list[str] | None) -> None:
    """Conserve the complete actual order when executing the previously collected cohort."""
    if declared is not None and nodes != declared:
        raise RuntimeError("Executed collection differs from the original complete cohort")


def validate_collect_only_result(status: int, unchanged: bool, nodes: list[str]) -> None:
    """Require actual framework success and unchanged source for complete collection."""
    if status != 0 or not unchanged or len(nodes) != 630:
        raise RuntimeError("Complete unchanged ABI collection failed")


def validate_complete_execution(result: ABIExecution) -> None:
    """Require the real full outcome without failed, skipped or repeated original cases."""
    if (
        result["pytest_exit"] != 0
        or result["junit_cases"] != 630
        or result["rejected"]
        or result["other_reports"]
        or not result["source_unchanged"]
        or len(result["passed"]) != 630
        or len(set(result["passed"])) != 630
    ):
        raise RuntimeError("The complete unchanged ABI module did not pass every case")


class Membership:
    """Observe actual full collection and every completed or refused framework report."""

    def __init__(self, job: Path, copied: Path, collect_only: bool) -> None:
        """Start without predeclared successful calls or fabricated framework metadata."""
        self.job = job
        self.copied = copied
        self.collect_only = collect_only
        self.nodes: list[str] = []
        self.passed: list[str] = []
        self.other: list[dict[str, str]] = []
        self.origin: str | None = None

    def pytest_collection_finish(self, session: CollectedSession) -> None:
        """Refuse omitted, duplicated or differently sourced actual collection members."""
        self.nodes = [MODULE + "::" + item.nodeid.split("::", 1)[1] for item in session.items]
        validate_node_count(self.nodes)
        module = sys.modules["tests.test_mat_family_c_abi"]
        self.origin = module.__file__
        validate_collection_origin(self.origin, self.copied)
        declared = (
            None if self.collect_only else json.loads((self.job / "collection.json").read_text())
        )
        validate_collection_order(self.nodes, declared)

    def pytest_runtest_logreport(self, report: CompletedReport) -> None:
        """Retain successful calls and every genuine setup failure, test failure or skip."""
        if report.when == "call" and report.passed:
            self.passed.append(MODULE + "::" + report.nodeid.split("::", 1)[1])
        elif report.failed or report.skipped:
            self.other.append(
                {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome}
            )
