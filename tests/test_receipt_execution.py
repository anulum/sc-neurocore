# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Validator execution evidence contract tests

"""Recorded test identities and results must support the claimed evidence."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.receipt_execution import (
    EXECUTION_CONTRACT,
    evidence_selection_problems,
    execution_problems,
    junit_checks,
    pytest_command,
)


def test_direct_command_detection_rejects_incidental_pytest_argument() -> None:
    """A Python snippet mentioning pytest is not a pytest execution."""
    assert pytest_command(["python3", "-m", "pytest", "tests/test_model.py"])
    assert pytest_command(["/venv/bin/pytest", "tests/test_model.py"])
    assert not pytest_command(["python", "-c", "pass", "pytest"])
    assert not pytest_command(["sh", "-c", "pytest"])
    assert not pytest_command([])


def test_evidence_selection_cannot_broaden_a_reviewed_test_node() -> None:
    """An exact node declaration permits neither its entire file nor a neighbour."""
    selected = "tests/test_receipt_execution.py::test_direct_command_detection_rejects_incidental_pytest_argument"
    assert not evidence_selection_problems([selected], selected)
    assert not evidence_selection_problems([selected], "tests/test_receipt_execution.py")
    assert evidence_selection_problems(["tests/test_receipt_execution.py"], selected)
    assert evidence_selection_problems([selected + "_other"], selected)
    assert evidence_selection_problems([selected], "")


def test_validator_nodes_not_suite_totals_determine_credit() -> None:
    """Inflated suite totals cannot substitute for actual testcase results."""
    report = '<testsuite tests="999"><testcase classname="tests.test_receipt_execution" name="test_direct_command_detection_rejects_incidental_pytest_argument" /></testsuite>'
    counts, nodes = junit_checks(report)
    assert counts["passed"] == counts["collected"] == len(nodes) == 1
    validator = {"execution_contract": EXECUTION_CONTRACT, "junit_xml": report}
    command = ["python", "-m", "pytest", "tests/test_receipt_execution.py"]
    assert not execution_problems(command, ["tests/test_receipt_execution.py"], validator, counts)
    assert execution_problems(
        command, ["tests/test_reference_lapicque_source_receipt.py"], validator, counts
    )
    assert execution_problems(command, [], validator, counts)
    assert execution_problems(
        command, ["tests/test_receipt_execution.py"], validator, {**counts, "passed": 999}
    )
    assert execution_problems(command, ["tests/test_receipt_execution.py"], {}, counts)


@pytest.mark.parametrize(
    "report", ["", "<wrong/>", "<testsuite/>", '<testsuite><testcase name="x" /></testsuite>']
)
def test_incomplete_reports_cannot_credit(report: str) -> None:
    """An absent validator identity is an error, not an anonymous passed check."""
    with pytest.raises(ValueError):
        junit_checks(report)


def test_parameterised_nodes_and_failure_categories_remain_explicit() -> None:
    """The unparameterised selector covers its instances without losing skips."""
    xml = '<testsuite><testcase classname="tests.test_receipt_execution" name="test_incomplete_reports_cannot_credit[a]" /><testcase classname="tests.test_receipt_execution" name="bad"><failure/></testcase><testcase classname="tests.test_receipt_execution" name="err"><error/></testcase><testcase classname="tests.test_receipt_execution" name="skip"><skipped/></testcase></testsuite>'
    counts, _nodes = junit_checks(xml)
    assert counts == {"collected": 4, "passed": 1, "failed": 1, "errors": 1, "skipped": 1}
    assert not execution_problems(
        ["python", "-m", "pytest"],
        ["tests/test_receipt_execution.py::test_incomplete_reports_cannot_credit"],
        {"execution_contract": EXECUTION_CONTRACT, "junit_xml": xml},
        counts,
    )
