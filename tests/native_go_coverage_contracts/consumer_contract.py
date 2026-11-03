# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual framework refusal and unchanged ABI custody contracts

"""Use real framework failures and damaged actual ABI records without inventing Pytest metadata."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import cast

import pytest

from tests.native_go_coverage_consumer import MODULE
from tools.go_coverage_consumer_inputs import (
    ABIExecution,
    validate_collect_only_result,
    validate_collection_order,
    validate_collection_origin,
    validate_complete_execution,
    validate_node_count,
    validate_source_projection,
)
from tools.go_mat_coverage_build import NativeGo


@pytest.fixture
def actual_adverse_reports(native_builder: NativeGo) -> dict[str, object]:
    """Obtain genuine setup-failure, call-failure and skip reports without running ABI models."""
    directory = native_builder.job / "artifacts/adverse-framework"
    directory.mkdir()
    source = directory / "test_actual_refusals.py"
    source.write_text(
        "import pytest\n"
        "@pytest.fixture\ndef rejected_setup():\n raise RuntimeError('actual setup refusal')\n"
        "def test_actual_setup_failure(rejected_setup):\n pass\n"
        "def test_actual_call_failure():\n raise RuntimeError('actual call refusal')\n"
        "def test_actual_skip():\n pytest.skip('actual skip control')\n"
    )
    output = directory / "actual-reports.json"
    driver = directory / "reports-driver.py"
    driver.write_text(
        "import json,sys,xml.etree.ElementTree as ET;from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);import pytest;"
        "from tools.go_coverage_consumer_inputs import Membership;"
        "directory=Path(sys.argv[2]);observer=Membership(directory,directory/'test_actual_refusals.py',True)\n"
        "class ReportsOnly:\n"
        " def pytest_runtest_logreport(self,report):\n  observer.pytest_runtest_logreport(report)\n"
        "status=pytest.main(['-q','-o','addopts=','-o','pythonpath=','-p','no:cacheprovider',"
        "'--rootdir',str(directory),'--confcutdir',str(directory),"
        "'--basetemp',str(directory/'tmp'),'--junitxml',str(directory/'actual.xml'),"
        "str(directory/'test_actual_refusals.py')],plugins=[ReportsOnly()])\n"
        "cases=ET.parse(directory/'actual.xml').findall('.//testcase')\n"
        "if status!=1 or len(cases)!=3:\n"
        " raise RuntimeError('Actual adverse framework cohort differs')\n"
        "result={'pytest_exit':int(status),'cases':[dict(c.attrib) for c in cases],"
        "'rejected':[dict(c.attrib) for c in cases if any(c.find(t) is not None "
        "for t in ('failure','error','skipped'))],'other_reports':observer.other,"
        "'original_ABI_cases_claimed':0}\n"
        "with Path(sys.argv[3]).open('x') as stream:\n json.dump(result,stream)\n"
    )
    assert (
        native_builder.command(
            "actual-adverse-framework",
            native_builder.python_script(
                driver, [str(native_builder.repository), str(directory), str(output)]
            ),
            native_builder.job,
            environment={**native_builder.environment, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        == 0
    )
    result = cast(dict[str, object], json.loads(output.read_text()))
    reports = cast(list[dict[str, str]], result["other_reports"])
    assert {(row["when"], row["outcome"]) for row in reports} == {
        ("setup", "failed"),
        ("call", "failed"),
        ("call", "skipped"),
    }
    assert result["original_ABI_cases_claimed"] == 0
    return result


def actual_changed_source_copy(repository: Path, output: Path) -> bool:
    """Measure a real change to an owned test-file copy while keeping the original source intact."""
    source = repository / MODULE
    copied = output / "changed-original-copy.py"
    shutil.copyfile(source, copied)
    before = hashlib.sha256(copied.read_bytes()).hexdigest()
    with copied.open("ab") as stream:
        stream.write(b"\n")
    return before == hashlib.sha256(copied.read_bytes()).hexdigest()


def test_actual_optimized_consumer_refuses_before_registration(native_builder: NativeGo) -> None:
    """The real interpreter optimization flag cannot admit disabled ABI assertions."""
    output = native_builder.job / "uncreated-consumer"
    output.mkdir()
    status = native_builder.command(
        "optimized-consumer",
        native_builder.python_script(
            native_builder.repository / "tests/native_go_coverage_consumer.py",
            [
                "--job",
                str(output),
                "--repository",
                str(native_builder.repository),
                "--collect-only",
            ],
            optimized=True,
        ),
        native_builder.job,
    )
    assert status == 1
    assert (
        "Native ABI qualification requires unoptimized assertions"
        in (native_builder.job / "commands/optimized-consumer/stderr.log").read_text()
    )
    assert list(output.iterdir()) == []


def test_changed_actual_source_projection_refuses(native_repository: Path, tmp_path: Path) -> None:
    """A copied complete original module loses qualification after a real byte change."""
    copied = tmp_path / "test_mat_family_c_abi.py"
    original = native_repository / MODULE
    shutil.copyfile(original, copied)
    with copied.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(RuntimeError, match="module changed in the coverage projection"):
        validate_source_projection(original, copied)


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_damaged_actual_collection_cannot_claim_all630_members(
    native_job: Path, fault: str
) -> None:
    """Only copied genuine node IDs are omitted or repeated; no Pytest session is fabricated."""
    nodes = cast(list[str], json.loads((native_job / "collection.json").read_text()))
    if fault == "missing":
        nodes.pop()
    else:
        nodes[-1] = nodes[0]
    with pytest.raises(RuntimeError, match="must contain all630"):
        validate_node_count(nodes)


@pytest.mark.parametrize("fault", ["absent", "other-source"])
def test_collection_origin_refuses_absence_or_a_different_real_source(
    native_job: Path, native_repository: Path, fault: str
) -> None:
    """Missing origin data and the actual foreign source path cannot match the projection."""
    origin = None if fault == "absent" else str(native_repository / MODULE)
    with pytest.raises(RuntimeError, match="imported from another source"):
        validate_collection_origin(origin, native_job / "subject" / MODULE)


def test_damaged_actual_collection_order_cannot_change_execution(native_job: Path) -> None:
    """The same630 genuine node IDs remain inadmissible after changing their recorded order."""
    declared = cast(list[str], json.loads((native_job / "collection.json").read_text()))
    changed = list(reversed(declared))
    assert changed != declared and set(changed) == set(declared)
    with pytest.raises(RuntimeError, match="differs from the original complete cohort"):
        validate_collection_order(changed, declared)


def test_collect_only_guard_refuses_genuine_framework_failure_and_damaged_custody(
    native_job: Path,
    native_repository: Path,
    tmp_path: Path,
    actual_adverse_reports: dict[str, object],
) -> None:
    """A real Pytest failure or damaged source/node observations cannot satisfy collection."""
    nodes = cast(list[str], json.loads((native_job / "collection.json").read_text()))
    observed_unchanged = actual_changed_source_copy(native_repository, tmp_path)
    assert observed_unchanged is False
    for status, unchanged, cohort in [
        (cast(int, actual_adverse_reports["pytest_exit"]), True, nodes),
        (0, observed_unchanged, nodes),
        (0, True, nodes[:-1]),
    ]:
        with pytest.raises(RuntimeError, match="unchanged ABI collection failed"):
            validate_collect_only_result(status, unchanged, cohort)


def test_complete_execution_guard_refuses_each_damaged_actual_result_dimension(
    native_job: Path,
    native_repository: Path,
    tmp_path: Path,
    actual_adverse_reports: dict[str, object],
) -> None:
    """Contrary framework facts and damaged actual results cannot relabel the original full pass."""
    path = native_job / "execution.json"
    original = path.read_bytes()
    actual = cast(ABIExecution, json.loads(original))
    unchanged = actual_changed_source_copy(native_repository, tmp_path)
    for fault in ("status", "cases", "rejected", "other", "source", "missing", "duplicate"):
        changed = deepcopy(actual)
        if fault == "status":
            changed["pytest_exit"] = cast(int, actual_adverse_reports["pytest_exit"])
        elif fault == "cases":
            changed["junit_cases"] = len(cast(list[object], actual_adverse_reports["cases"]))
        elif fault == "rejected":
            changed["rejected"] = cast(list[dict[str, str]], actual_adverse_reports["rejected"])
        elif fault == "other":
            changed["other_reports"] = cast(
                list[dict[str, str]], actual_adverse_reports["other_reports"]
            )
        elif fault == "source":
            changed["source_unchanged"] = unchanged
        elif fault == "missing":
            changed["passed"].pop()
        else:
            changed["passed"][-1] = changed["passed"][0]
        with pytest.raises(RuntimeError, match="did not pass every case"):
            validate_complete_execution(changed)
    assert path.read_bytes() == original
