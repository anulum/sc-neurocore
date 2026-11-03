# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Refusals of damaged actual native build observations

"""Use genuine formatter output and damaged copies of actual symbols, dependencies and tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import cast

import pytest

from tools.go_coverage_build_inputs import (
    validate_gofmt_output,
    validate_native_dependencies,
    validate_native_exports,
    validate_native_test_events,
    validate_native_test_names,
)
from tools.go_coverage_profiles import MODULE
from tools.go_mat_coverage_build import NativeGo


def actual_command_output(job: Path, label: str) -> str:
    """Read only the original actual tool stdout retained by this admitted native producer."""
    return (job / "commands" / label / "stdout.log").read_text()


def test_real_formatter_refuses_an_owned_unformatted_copy(
    native_builder: NativeGo, tmp_path: Path
) -> None:
    """The pinned SDK formatter itself lists the damaged real source before validation refuses."""
    text = (native_builder.source / "services/mat_simulation_coverage_test.go").read_text()
    assert "package services" in text
    source = tmp_path / "unformatted.go"
    source.write_text(text.replace("package services", "package  services", 1))
    native_builder.require(
        "actual-unformatted-go",
        [str(native_builder.go.parent / "gofmt"), "-l", str(source)],
        tmp_path,
    )
    output = actual_command_output(native_builder.job, "actual-unformatted-go")
    assert str(source) in output
    with pytest.raises(RuntimeError, match="not formatted"):
        validate_gofmt_output(output)


@pytest.mark.parametrize("model", ["mat", "sc_resetting_mat"])
@pytest.mark.parametrize("fault", ["simulation", "unexpected-diagnostic", "missing-diagnostic"])
def test_damaged_actual_symbol_table_cannot_cross_the_build_tag_boundary(
    native_job: Path, model: str, fault: str
) -> None:
    """Change copied symbol rows only; both actual ordinary and tagged native images stay intact."""
    variant = "tagged-uninstrumented" if fault == "missing-diagnostic" else "normal"
    rows = actual_command_output(native_job, "symbols-" + variant + "-" + model).splitlines()
    if fault == "simulation":
        rows = [line for line in rows if line.split()[-1] != model + "_simulate_c"]
    elif fault == "missing-diagnostic":
        rows = [line for line in rows if "coverage_snapshot" not in line]
    else:
        tagged = actual_command_output(native_job, "symbols-tagged-uninstrumented-" + model)
        rows.append(next(line for line in tagged.splitlines() if "coverage_snapshot" in line))
    with pytest.raises(RuntimeError, match="simulation export|build tag"):
        validate_native_exports(model, variant, "\n".join(rows))


@pytest.mark.parametrize("tagged", [False, True])
@pytest.mark.parametrize("package", [MODULE + "internal/coverageprofile", "runtime/coverage"])
def test_damaged_actual_dependency_list_cannot_enable_or_hide_coverage_packages(
    native_job: Path, tagged: bool, package: str
) -> None:
    """Transfer or remove a genuine observed dependency in a copy without replacing the compiler."""
    label = "tagged-dependencies" if tagged else "normal-dependencies"
    rows = actual_command_output(native_job, label).splitlines()
    assert package in actual_command_output(native_job, "tagged-dependencies").splitlines()
    if tagged:
        rows.remove(package)
    else:
        rows.append(package)
    with pytest.raises(RuntimeError, match="build tag"):
        validate_native_dependencies(tagged, rows)


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_damaged_actual_native_test_names_cannot_change_the_full_source_cohort(
    native_job: Path, fault: str
) -> None:
    """The completed source declaration remains fixed while copied names lose or repeat a member."""
    membership = json.loads((native_job / "artifacts/native-membership.json").read_text())
    names = cast(list[str], membership["expected"]).copy()
    if fault == "missing":
        names.pop()
    else:
        names[-1] = names[0]
    with pytest.raises(RuntimeError, match="cohort changed"):
        validate_native_test_names(names)


@pytest.mark.parametrize("fault", ["fail", "skip", "missing", "duplicate", "name", "name-type"])
def test_damaged_real_go_events_cannot_claim_a_complete_native_pass(
    native_job: Path, fault: str
) -> None:
    """Corrupt copied actual events and require refusal without replaying numerical work."""
    membership = json.loads((native_job / "artifacts/native-membership.json").read_text())
    expected = cast(list[str], membership["expected"])
    events: list[dict[str, object]] = [
        json.loads(line)
        for line in actual_command_output(native_job, "native-family-tests").splitlines()
    ]
    passed = [event for event in events if event.get("Action") == "pass" and "Test" in event]
    if fault in {"fail", "skip"}:
        passed[0]["Action"] = fault
    elif fault == "missing":
        events.remove(passed[0])
    elif fault == "duplicate":
        events.append(deepcopy(passed[0]))
    elif fault == "name":
        passed[0]["Test"] = passed[1]["Test"]
    else:
        passed[0]["Test"] = [passed[0]["Test"]]
    with pytest.raises(RuntimeError, match="failed or skipped|full family cohort"):
        validate_native_test_events(expected, events)
