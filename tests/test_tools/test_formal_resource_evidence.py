# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual native resource-incomplete evidence contracts

"""Exercise real exhausted solvers and refuse tampered resource evidence."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from tests.formal_native_evidence_support import JOB, NativeEvidence
from tests.formal_native_resource_support import bind_resource_row, collect_native_resources
from tools.formal_evidence_validation import digest, object_fields, read_object
from tools.formal_resource_evidence import validate_resource_attempt
from tools.formal_external_evidence import main, validate_external


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> NativeEvidence:
    """Collect actual bounded native attempts once for this module's controls."""
    return collect_native_resources(tmp_path_factory.mktemp("native-resource-row"))


@pytest.fixture
def case(native: NativeEvidence, tmp_path: Path) -> NativeEvidence:
    """Isolate small real inputs and receipts without duplicating solver output."""
    destination = tmp_path / JOB.parent
    destination.mkdir(parents=True)
    for source in (native.repository / JOB.parent).iterdir():
        shutil.copyfile(source, destination / source.name)
    for source in native.repository.iterdir():
        if source.is_file():
            shutil.copyfile(source, tmp_path / source.name)
    return replace(native, repository=tmp_path)


def _check(case: NativeEvidence, entry: dict[str, object] | None = None) -> dict[str, object]:
    """Exercise the public receipt validator with the real local resource policy."""
    return validate_external(
        case.repository,
        JOB,
        bind_resource_row(case) if entry is None else entry,
        case.provenance,
        case.solvers,
        policy=case.policy,
    )


def _change(path: Path, field: str, value: object) -> None:
    """Mutate one actual artifact field while preserving the other native facts."""
    data = read_object(path)
    parent = data
    parts = field.split(".")
    for part in parts[:-1]:
        child = object_fields(parent[part])
        parent[part] = child
        parent = child
    parent[parts[-1]] = value
    path.write_text(json.dumps(data))


def test_actual_dual_solver_timeout_row_is_never_pass(case: NativeEvidence) -> None:
    """Two actual native resource stops plus reached enables remain UNPROVEN."""
    result = _check(case)
    assert result["native_status"] == "UNPROVEN-RESOURCE"
    assert object_fields(result["nonvacuity"])["status"] == "NONVACUOUS"
    assert result["depth_scope"] == {"default": {"mode": "bmc", "depth": 1000000}}
    assert float(str(result["wall_seconds"])) >= 2 * case.policy.timeout_seconds
    with pytest.raises(ValueError, match="host_profile"):
        validate_external(
            case.repository, JOB, bind_resource_row(case), case.provenance, case.solvers
        )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("schema_version", True, "schema"),
        ("measurement.status", "ERROR", "not TIMEOUT or MEMORY"),
        ("measurement.status", "UNKNOWN", "not TIMEOUT or MEMORY"),
        ("measurement.status", "FAIL", "not TIMEOUT or MEMORY"),
        ("measurement.status", "PASS", "not TIMEOUT or MEMORY"),
        ("measurement.host_profile", "unadmitted-profile", "host_profile"),
        ("measurement.unit_accounting", "missing", "unit_accounting"),
        ("measurement.native_exit_source", "log", "native_exit_source"),
        ("measurement.wall_clock_source", "estimated", "wall_clock_source"),
        ("measurement.active_state", "active", "active_state"),
        ("measurement.cgroup_processes", [123], "cgroup_processes"),
        ("measurement.process_tree_closed", 1, "process tree"),
        ("measurement.main_pid", True, "main_pid"),
        ("measurement.memory_limit_bytes", 64 * 1024**3, "memory_limit_bytes"),
        ("measurement.swap_limit_bytes", 1, "swap_limit_bytes"),
        ("measurement.time_limit_seconds", 7200, "time_limit_seconds"),
        ("measurement.invocation_id", None, "invocation identity"),
        ("measurement.invocation_id", "wrong", "invocation identity"),
        ("measurement.native_exit_code", False, "exit codes"),
        ("measurement.native_exit_code", 0, "exit codes"),
        ("measurement.unit_exit_code", 123, "exit codes"),
        ("measurement.unit_exec_code", 3, "exit kind"),
        ("measurement.unit_exec_code", True, "exit kind"),
        ("measurement.unit_result", "exit-code", "stop cause"),
        ("measurement.start_monotonic_microseconds", 0, "clock stamps"),
        ("measurement.stop_monotonic_microseconds", False, "clock stamps"),
        ("measurement.wall_seconds", 0, "monotonic stamps"),
        ("measurement.peak_memory_bytes", 0, "peak is outside"),
        ("measurement.peak_memory_bytes", 128 * 1024**2 + 1, "peak is outside"),
        ("measurement.peak_memory_bytes", 1.5, "peak is outside"),
        ("measurement.oom_events.oom", -1, "nonnegative"),
        ("measurement.oom_events.oom", True, "nonnegative"),
        ("measurement.oom_events.oom_kill", 1, "also reports an OOM"),
        ("task_statuses", {}, "inventory"),
        ("task_statuses.default", "ERROR", "task reports"),
        ("task_statuses.default", "FAIL", "task reports"),
        ("task_statuses.default", "UNKNOWN", "task reports"),
        ("task_statuses.default", "PASS", "completed proof"),
    ],
)
def test_changed_actual_resource_fact_refuses(
    case: NativeEvidence, field: str, value: object, reason: str
) -> None:
    """Refuse one tampered native fact after rebinding the test-only outer review."""
    _change(case.repository / "receipt.json", field, value)
    with pytest.raises(ValueError, match=reason):
        _check(case)


@pytest.mark.parametrize(
    "marker,reason",
    [
        ("DONE (FAIL, rc=2)", "proof failure or tool error"),
        ("DONE (ERROR, rc=16)", "proof failure or tool error"),
        ("DONE (UNKNOWN, rc=1)", "proof failure or tool error"),
        ("DONE (PASS, rc=1)", "proof failure or tool error"),
        ("ERROR: tool failed", "tool failure or counterexample"),
        ("Traceback (most recent call last):", "tool failure or counterexample"),
        ("Illegal instruction", "tool failure or counterexample"),
        ("Segmentation fault", "tool failure or counterexample"),
        ("Broken pipe", "tool failure or counterexample"),
        ("Assert failed", "tool failure or counterexample"),
        ("Status: unknown", "tool failure or counterexample"),
    ],
)
def test_resource_log_tool_failure_cannot_be_relabelled(
    case: NativeEvidence, marker: str, reason: str
) -> None:
    """A rehashed native log with an actual failure marker cannot qualify a row."""
    path = case.repository / "primary-public.log"
    path.write_text(path.read_text() + "\n" + marker + "\n")
    _change(case.repository / "receipt.json", "log.normalized_sha256", digest(path))
    with pytest.raises(ValueError, match=reason):
        _check(case)


@pytest.mark.parametrize("attempts", [None, {}, [], [None, None, None, None]])
def test_incomplete_or_unbounded_history_refuses(case: NativeEvidence, attempts: object) -> None:
    """Require a finite explicit admitted chain through the public row entry."""
    entry = bind_resource_row(case)
    entry["attempts"] = attempts
    with pytest.raises(ValueError, match="complete admitted attempt history"):
        _check(case, entry)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("verdict", "PENDING", "verdict binding"),
        ("schema_version", 2, "schema"),
        ("outcome", "PASS", "outcome binding"),
        ("job", "hdl/formal/other.sby", "job binding"),
        ("attempts", [], "attempts binding"),
        ("nonvacuity_sha256", "0" * 64, "nonvacuity_sha256 binding"),
        ("nonvacuity_review_sha256", "0" * 64, "nonvacuity_review_sha256 binding"),
        ("depth_scope", {}, "depth_scope binding"),
        ("reviewer", "", "distinct identified"),
    ],
)
def test_row_review_must_bind_all_actual_evidence(
    case: NativeEvidence, field: str, value: object, reason: str
) -> None:
    """Refuse a changed review after updating its outer test-only digest."""
    entry = bind_resource_row(case)
    path = case.repository / "row-review.json"
    _change(path, field, value)
    entry["review_sha256"] = digest(path)
    with pytest.raises(ValueError, match=reason):
        _check(case, entry)


def test_audit_cannot_silently_switch_to_the_other_solver_parent(case: NativeEvidence) -> None:
    """The real Z3 enable audit cannot be rebound to the actual Yices receipt."""
    entry = bind_resource_row(case)
    attempts = entry["attempts"]
    assert isinstance(attempts, list)
    entry["nonvacuity_parent_sha256"] = object_fields(attempts[1])["receipt_sha256"]
    with pytest.raises(ValueError, match="another primary receipt"):
        _check(case, entry)


def test_one_solver_alone_is_not_dual_solver_exhaustion(case: NativeEvidence) -> None:
    """A genuine Z3 timeout alone cannot satisfy the final UNPROVEN contract."""
    entry = bind_resource_row(case)
    attempts = entry["attempts"]
    assert isinstance(attempts, list)
    entry["attempts"] = attempts[:1]
    with pytest.raises(ValueError, match="final actual Yices attempt"):
        _check(case, entry)


def test_actual_cli_dispatch_preserves_unproven_status(
    case: NativeEvidence, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real argument parser and dispatch expose the qualified incomplete row."""
    entry = bind_resource_row(case)
    index = case.repository / "index.json"
    index.write_text(json.dumps({"schema_version": 1, "jobs": {JOB.as_posix(): entry}}))
    provenance = case.repository / "PROVENANCE.txt"
    provenance.write_text(case.provenance)
    result = main(
        [
            JOB.as_posix(),
            "--repository",
            str(case.repository),
            "--index",
            index.name,
            "--provenance",
            str(provenance),
        ],
        policy=case.policy,
    )
    output = capsys.readouterr()
    assert result == 0, output.err
    assert object_fields(json.loads(output.out))["native_status"] == "UNPROVEN-RESOURCE"


@pytest.mark.parametrize("position,status", [(0, "MEMORY"), (1, "TIMEOUT")])
def test_actual_prove_lane_resource_predecessors(position: int, status: str) -> None:
    """Recognise actual SMTBMC basecase and induction solver-start records."""
    repository = Path(__file__).resolve().parents[2]
    index = object_fields(read_object(repository / "benchmarks/results/formal/index.json")["jobs"])
    job = Path("hdl/formal/catalogue/sc_chaotic_map.sby")
    history = object_fields(index[job.as_posix()])["predecessors"]
    assert isinstance(history, list)
    entry = object_fields(history[position])
    yosys, z3 = shutil.which("yosys"), shutil.which("z3")
    assert yosys is not None and z3 is not None
    provenance = (Path(yosys).resolve().parents[1] / "PROVENANCE.txt").read_text()
    summary, receipt = validate_resource_attempt(
        repository, job, entry, provenance, {"z3": Path(z3)}
    )
    assert summary["native_status"] == status
    assert object_fields(receipt["measurement"])["status"] == status
