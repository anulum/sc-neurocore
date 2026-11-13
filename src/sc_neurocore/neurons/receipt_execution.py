# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed validator evidence for readiness receipts

"""Bind credited checks to retained pytest results, not process exit status."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

EXECUTION_CONTRACT = "pytest-junit.v1"
#: Test case property naming the backend lane a run executed.
LANE_PROPERTY = "sc_neurocore.lane"
#: Test case property naming one native library a run loaded, as ``<file name>=<sha256>``.
NATIVE_BINARY_PROPERTY = "sc_neurocore.native_binary"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def evidence_selection_problems(selected: Sequence[str], declared: str) -> tuple[str, ...]:
    """Reject validator selections outside a descriptor's reviewed evidence.

    Parameters
    ----------
    selected : Sequence[str]
        Evidence references selected for this particular execution.
    declared : str
        Descriptor evidence field containing reviewed files or exact test nodes.

    Returns
    -------
    tuple[str, ...]
        Missing or foreign test selections. A file declaration permits its nodes,
        but a node declaration does not permit another test in the same file.
    """
    allowed = [part.strip() for part in declared.split(";") if part.strip().startswith("tests/")]
    requested = [
        part.strip()
        for field in selected
        for part in field.split(";")
        if part.strip().startswith("tests/")
    ]
    if not allowed or not requested:
        return ("no descriptor-bound executable validator selection",)
    return tuple(
        f"validator is outside declared evidence: {node}"
        for node in requested
        if not any(
            node == item or (item.endswith(".py") and node.startswith(item + "::"))
            for item in allowed
        )
    )


def pytest_command(command: Sequence[str]) -> bool:
    """Return whether argv directly invokes pytest, without a shell wrapper.

    Parameters
    ----------
    command : Sequence[str]
        Executable and arguments.

    Returns
    -------
    bool
        Whether pytest owns the invocation rather than appearing as an argument.
    """
    if not command:
        return False
    executable = Path(command[0]).name
    return executable in {"pytest", "pytest.exe"} or (
        executable.startswith("python") and tuple(command[1:3]) == ("-m", "pytest")
    )


def junit_checks(xml: str) -> tuple[dict[str, int], tuple[str, ...]]:
    """Derive counts and executed validator names from the actual report.

    Parameters
    ----------
    xml : str
        Complete pytest JUnit XML retained in a receipt.

    Returns
    -------
    tuple[dict[str, int], tuple[str, ...]]
        Counts derived from testcase elements and their dotted identities.

    Raises
    ------
    ValueError
        If the report is malformed, empty, declares a DTD or entities, or lacks
        testcase identities. External references are never resolved.
    """
    counts = dict.fromkeys(("collected", "passed", "failed", "errors", "skipped"), 0)
    nodes: list[str] = []
    for case in _report_root(xml).iter("testcase"):
        nodes.append(_case_identity(case))
        counts["collected"] += 1
        status = "passed"
        for tag, key in (("error", "errors"), ("failure", "failed"), ("skipped", "skipped")):
            if case.find(tag) is not None:
                status = key
                break
        counts[status] += 1
    if not nodes:
        raise ValueError("JUnit report has no executed checks")
    return counts, tuple(nodes)


def _report_root(xml: str) -> Any:
    """Parse a JUnit report without resolving a DTD, entities or external references."""
    try:
        root = ET.fromstring(xml, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except (ET.ParseError, DefusedXmlException) as error:
        raise ValueError("invalid JUnit report") from error
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError("not a JUnit report")
    return root


def _case_identity(case: Any) -> str:
    """Return the dotted validator identity of one test case element."""
    name, owner = case.get("name", ""), case.get("classname", "")
    if not name or not owner:
        raise ValueError("JUnit testcase lacks its validator identity")
    return f"{owner}.{name}"


@dataclass(frozen=True)
class CaseLane:
    """The lane one executed test case ran and the native libraries it loaded.

    Attributes
    ----------
    case : str
        Dotted validator identity of the test case.
    lane : str
        The backend lane the case reports.
    libraries : tuple of tuple of str
        ``(file name, sha256)`` of each library, in report order, each once.
    """

    case: str
    lane: str
    libraries: tuple[tuple[str, str], ...]


def junit_lane_evidence(xml: str) -> tuple[tuple[CaseLane, ...], tuple[str, ...]]:
    """Return the lane evidence of every executed test case and every inconsistency.

    Parameters
    ----------
    xml : str
        Complete pytest JUnit XML retained in a receipt.

    Returns
    -------
    tuple
        The executed cases that report exactly one lane, in report order, and the
        problems: a case that reports no lane or more than one, a library property
        that is not ``<file name>=<sha256>``, and one library name of one lane with
        two digests in the report. Two lanes may load libraries of the same name
        (the Go and the Mojo lane of a model each load their own
        ``lib<model>.so``), so a name is compared within its lane only, and the
        libraries of a case that reports no single lane are not compared. A
        skipped case is not evidence and is not checked.

    Raises
    ------
    ValueError
        As :func:`junit_checks`: a malformed report, a declared DTD or entities, a
        foreign root, or a test case without its identity.
    """
    cases: list[CaseLane] = []
    problems: list[str] = []
    digests: dict[tuple[str, str], str] = {}
    for case in _report_root(xml).iter("testcase"):
        identity = _case_identity(case)
        if case.find("skipped") is not None:
            continue
        lanes: list[str] = []
        libraries: list[tuple[str, str]] = []
        block = case.find("properties")
        for item in block.iter("property") if block is not None else ():
            name, value = item.get("name", ""), item.get("value", "")
            if name == LANE_PROPERTY:
                lanes.append(value)
            elif name == NATIVE_BINARY_PROPERTY:
                file, separator, digest = value.rpartition("=")
                if not separator or not file or not _SHA256.fullmatch(digest):
                    problems.append(f"test case {identity} reports a malformed library {value!r}")
                    continue
                libraries.append((file, digest))
        distinct = list(dict.fromkeys(lanes))
        if not distinct:
            problems.append(f"test case {identity} reports no lane")
        elif len(distinct) > 1:
            problems.append(
                f"test case {identity} reports more than one lane: {', '.join(sorted(distinct))}"
            )
        else:
            lane = distinct[0]
            for file, digest in libraries:
                known = digests.setdefault((lane, file), digest)
                conflict = f"library {file} of lane {lane} has two digests in one report: {known}, {digest}"
                if known != digest and conflict not in problems:
                    problems.append(conflict)
            cases.append(CaseLane(identity, lane, tuple(dict.fromkeys(libraries))))
    return tuple(cases), tuple(problems)


def execution_problems(
    command: Sequence[str],
    evidence_refs: Sequence[str],
    validator: Mapping[str, str],
    counts: Mapping[str, int],
) -> tuple[str, ...]:
    """Check that a receipt's declared validators actually appear in its run.

    Parameters
    ----------
    command : Sequence[str]
        Recorded direct invocation.
    evidence_refs : Sequence[str]
        Declared test files or node IDs; prose alone cannot identify a validator.
    validator : Mapping[str, str]
        Execution contract and retained JUnit XML.
    counts : Mapping[str, int]
        Recorded check totals, compared against the XML testcase elements.

    Returns
    -------
    tuple[str, ...]
        Reasons why this execution cannot credit the scientific claim.
    """
    if validator.get("execution_contract") != EXECUTION_CONTRACT or not pytest_command(command):
        return ("execution lacks a supported validator-result contract",)
    try:
        observed, nodes = junit_checks(validator.get("junit_xml", ""))
    except ValueError as error:
        return (str(error),)
    problems = []
    if dict(counts) != observed:
        problems.append("recorded counts disagree with executed JUnit checks")
    expected = []
    for field in evidence_refs:
        for raw in field.split(";"):
            path, separator, node = raw.strip().partition("::")
            if path.startswith("tests/") and path.endswith(".py"):
                prefix = path[:-3].replace("/", ".") + "."
                expected.append((prefix + node.replace("::", ".")) if separator else prefix)
    if not expected:
        problems.append("no declared executable test validator")
    for name in expected:
        if not any(
            candidate.startswith(name)
            if name.endswith(".")
            else (candidate == name or candidate.startswith(name + "["))
            for candidate in nodes
        ):
            problems.append(f"declared validator was not executed: {name}")
    return tuple(problems)
