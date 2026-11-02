# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound external assertion-enable audit controls

"""Accept actual separately measured audits and reject altered evidence bindings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.formal_native_evidence_support import JOB, NativeEvidence, collect_native, copy_case
from tools.formal_evidence_validation import digest, object_fields, read_object
from tools.formal_external_evidence import validate_external
from tools.formal_job_inputs import resolve_job
from tools.formal_nonvacuity_evidence import validate_nonvacuity


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> NativeEvidence:
    """Collect a real primary and an independently bounded native enable audit."""
    return collect_native(tmp_path_factory.mktemp("native-external-enable"))


@pytest.fixture
def case(native: NativeEvidence, tmp_path: Path) -> NativeEvidence:
    """Keep original native measurements immutable across refusal controls."""
    return copy_case(native, tmp_path)


def _check(case: NativeEvidence, entry: dict[str, object]) -> dict[str, object]:
    """Exercise the public artifact validator with the actual primary receipt."""
    return validate_nonvacuity(
        case.repository,
        resolve_job(case.repository, JOB),
        entry,
        read_object(case.repository / "receipt.json"),
        digest(case.repository / "receipt.json"),
        ["default"],
        policy=case.policy,
    )


def _mutate(case: NativeEvidence, field: str, value: object) -> None:
    """Alter one field while preserving the rest of an actual native artifact."""
    path = case.repository / "nonvacuity.json"
    data = read_object(path)
    parent = data
    parts = field.split(".")
    for part in parts[:-1]:
        child = object_fields(parent[part])
        parent[part] = child
        parent = child
    parent[parts[-1]] = value
    path.write_text(json.dumps(data))


def _refresh(case: NativeEvidence, entry: dict[str, object]) -> None:
    """Rebind only outer test metadata so the targeted inner guard is exercised."""
    path = case.repository / "nonvacuity.json"
    entry["nonvacuity_sha256"] = digest(path)
    review_path = case.repository / "nonvacuity-review.json"
    review = read_object(review_path)
    review["artifact_sha256"] = digest(path)
    review_path.write_text(json.dumps(review))
    entry["nonvacuity_review_sha256"] = digest(review_path)


def test_actual_separate_audit_and_external_gate(case: NativeEvidence) -> None:
    """Require a separate actual measurement while preserving the primary bytes."""
    before = digest(case.repository / "receipt.json")
    entry = case.rebind()
    result = validate_external(
        case.repository, JOB, entry, case.provenance, case.solvers, policy=case.policy
    )
    assert object_fields(result["nonvacuity"])["status"] == "NONVACUOUS"
    audit = read_object(case.repository / "nonvacuity.json")
    primary = read_object(case.repository / "receipt.json")
    assert (
        object_fields(audit["measurement"])["invocation_id"]
        != object_fields(primary["measurement"])["invocation_id"]
    )
    assert digest(case.repository / "receipt.json") == before


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("schema_version", True, "schema"),
        ("receipt_sha256", "0" * 64, "another primary"),
        ("source_hashes", {}, "sources are stale"),
        ("toolchain", {}, "tools differ"),
        ("measurement.unit_accounting", "garbage-collected", "not retained"),
        ("measurement.status", "TIMEOUT", "status"),
        ("measurement.process_tree_closed", False, "process tree"),
        ("measurement.memory_limit_bytes", 1, "memory_limit"),
        ("tasks", {}, "every primary task"),
        ("tasks.default.configuration", {}, "configuration differs"),
        ("tasks.default.report.depth", True, "mode or depth"),
        ("tasks.default.report.mode", "prove", "mode or depth"),
        ("tasks.default.report.schema_version", 2, "not NONVACUOUS"),
        ("tasks.default.report.verdict", "ERROR", "not NONVACUOUS"),
        ("tasks.default.report.design_prep_sha256", "bad", "missing or invalid"),
        ("tasks.default.report.sby_sha256", "0" * 64, "SBY binary differs"),
        ("tasks.default.model_custody.audited_sha256", "0" * 64, "Audited model"),
        ("tasks.default.model_custody.primary_sha256", "0" * 64, "custody"),
        ("tasks.default.model_custody.origin", "unknown", "custody"),
        ("tasks.default.report.implementation_sha256", "0" * 64, "implementation is stale"),
        ("tasks.default.report.assertions", {}, "identities"),
        ("tasks.default.report.first_reached_steps", {}, "UNREACHED"),
        ("tasks.default.report.assertion_enable_status", "UNREACHED", "UNREACHED"),
        ("tasks.default.report.unreached_assertions", ["one"], "UNREACHED"),
        ("tasks.default.report.transform_sha256", "0" * 64, "transform is stale"),
        ("tasks.default.report.native_exit_code", False, "cover did not PASS"),
        ("tasks.default.report.native_status", "FAIL", "cover did not PASS"),
        ("tasks.default.report.auxiliary_log_sha256", "0" * 64, "log binding"),
        ("producer", "", "distinct identified"),
    ],
)
def test_altered_actual_audit_is_refused(
    case: NativeEvidence, field: str, value: object, reason: str
) -> None:
    """A refreshed outer digest cannot hide a changed source, model or witness."""
    entry = case.rebind()
    _mutate(case, field, value)
    _refresh(case, entry)
    with pytest.raises(ValueError, match=reason):
        _check(case, entry)


@pytest.mark.parametrize("step", [-1, 4, True])
def test_witness_must_be_an_actual_step_inside_depth(case: NativeEvidence, step: object) -> None:
    """Reject negative, off-bound and boolean witness steps despite outer rebinding."""
    entry = case.rebind()
    _mutate(case, "tasks.default.report.first_reached_steps", {"scn_assertion_enable_000000": step})
    _refresh(case, entry)
    with pytest.raises(ValueError, match="UNREACHED"):
        _check(case, entry)


@pytest.mark.parametrize("field", ["unit", "invocation_id"])
def test_primary_invocation_cannot_be_reused_as_audit(case: NativeEvidence, field: str) -> None:
    """The audit has its own native service rather than borrowed primary accounting."""
    entry = case.rebind()
    primary = read_object(case.repository / "receipt.json")
    _mutate(case, "measurement." + field, object_fields(primary["measurement"])[field])
    _refresh(case, entry)
    with pytest.raises(ValueError, match="separate native service"):
        _check(case, entry)


@pytest.mark.parametrize("change", ["done", "witness"])
def test_actual_log_must_contain_the_native_witness_and_success(
    case: NativeEvidence, change: str
) -> None:
    """A rehashed changed native log cannot contradict its claimed enable result."""
    entry = case.rebind()
    path = case.repository / "public-audit.log"
    text = path.read_text()
    text = (
        text.replace("DONE (PASS, rc=0)", "DONE (FAIL, rc=2)")
        if change == "done"
        else "\n".join(line for line in text.splitlines() if "Reached cover statement" not in line)
    )
    path.write_text(text)
    _mutate(case, "tasks.default.log.normalized_sha256", digest(path))
    _refresh(case, entry)
    with pytest.raises(ValueError, match="DONE marker|lacks an enable witness"):
        _check(case, entry)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("verdict", "OPEN"),
        ("artifact_sha256", "0" * 64),
        ("receipt_sha256", "0" * 64),
        ("source_hashes", {}),
        ("prepared_model_sha256", {}),
        ("reviewer", ""),
        ("reviewer", "actual separately measured native enable-audit fixture"),
    ],
)
def test_independent_audit_review_bindings(case: NativeEvidence, field: str, value: object) -> None:
    """Require an identified independent reviewer and the exact artifact/model bindings."""
    entry = case.rebind()
    path = case.repository / "nonvacuity-review.json"
    review = read_object(path)
    review[field] = value
    path.write_text(json.dumps(review))
    entry["nonvacuity_review_sha256"] = digest(path)
    with pytest.raises(ValueError):
        _check(case, entry)


@pytest.mark.parametrize(
    "field", ["nonvacuity", "nonvacuity_sha256", "nonvacuity_review", "nonvacuity_review_sha256"]
)
def test_missing_audit_or_review_is_refused(case: NativeEvidence, field: str) -> None:
    """A primary PASS alone never substitutes for separately indexed audit evidence."""
    entry = case.rebind()
    entry.pop(field)
    with pytest.raises(ValueError):
        _check(case, entry)


def test_real_cover_mode_audit(tmp_path: Path) -> None:
    """Audit the same actual reset property as a native reached-cover task."""
    case = collect_native(tmp_path, cover_mode=True)
    result = _check(case, case.rebind())
    assert result["status"] == "NONVACUOUS"
    task = object_fields(
        object_fields(read_object(case.repository / "nonvacuity.json")["tasks"])["default"]
    )
    report = object_fields(task["report"])
    assert report["mode"] == "cover" and report["original_cover_goals"]


def test_declared_task_named_default_is_not_the_unnamed_task(tmp_path: Path) -> None:
    """Use the native conditional options for an actual task whose name is default."""
    case = collect_native(tmp_path, named_default=True)
    result = validate_external(
        case.repository, JOB, case.rebind(), case.provenance, case.solvers, policy=case.policy
    )
    assert object_fields(result["nonvacuity"])["status"] == "NONVACUOUS"
