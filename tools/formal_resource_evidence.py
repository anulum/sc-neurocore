# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reviewed dual-solver resource-incomplete evidence

"""Qualify explicitly reviewed resource-incomplete rows without claiming PASS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import re

from tools.formal_evidence_validation import (
    REFERENCE_HOST_PROFILE,
    EvidencePolicy,
    confined_file,
    digest,
    object_fields,
    read_object,
    require,
)
from tools.formal_job_inputs import resolve_job
from tools.formal_log_normalization import validate_public_log
from tools.formal_nonvacuity_evidence import validate_nonvacuity
from tools.formal_receipt_identity import validate_identity, validate_review
from tools.formal_resource_accounting import validate_resource_measurement
from tools.formal_resource_policy import YICES_RETRY, next_attempt, select_contract


def _log_line(log: str, offset: int) -> str:
    """Return the log line holding ``offset``, stripped and bounded to 200 characters."""
    start = log.rfind("\n", 0, offset) + 1
    return log[start:].split("\n", 1)[0].strip()[:200]


def validate_resource_attempt(
    repository: Path,
    job: Path,
    entry: dict[str, object],
    provenance: str,
    solvers: Mapping[str, Path],
    *,
    policy: EvidencePolicy | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate one actual incomplete attempt and its independent attestation.

    Parameters
    ----------
    repository : Path
        Resolved checkout with unchanged model and retained evidence.
    job : Path
        Current canonical formal configuration.
    entry : dict of str to object
        Explicit contract, receipt, review and any solver-only variant.
    provenance : str
        Current verified SDK provenance.
    solvers : mapping of str to Path
        Actual current solver executables.
    policy : EvidencePolicy or None, optional
        Actual bounded native fixture policy; the CLI uses fixed production limits.

    Returns
    -------
    tuple of dict of str to object and dict of str to object
        Validated outcome summary and unchanged original receipt fields.

    Raises
    ------
    ValueError
        If sources, tools, accounting, logs, contract or review do not qualify.
    OSError
        If a required native file or executable is unavailable.
    """
    receipt_path = confined_file(repository, entry.get("receipt"))
    receipt_sha = digest(receipt_path)
    require(receipt_sha == entry.get("receipt_sha256"), "Resource receipt digest differs.")
    receipt = read_object(receipt_path)
    require(
        type(receipt.get("schema_version")) is int and receipt.get("schema_version") == 1,
        "Unknown resource receipt schema.",
    )
    inputs, tasks, solver = validate_identity(repository, job, entry, receipt, provenance, solvers)
    review_path, review = validate_review(repository, entry, receipt, receipt_sha)
    contract = select_contract(
        entry, review, receipt_sha, inputs.job.relative_to(repository).as_posix()
    )
    require(solver == contract.solver, "Resource solver differs from the indexed contract.")
    status, wall, peak = validate_resource_measurement(
        receipt, contract.policy() if policy is None else policy
    )
    task_statuses = object_fields(receipt.get("task_statuses"))
    require(set(task_statuses) == set(tasks), "Resource task inventory differs.")
    require(
        all(
            value is None or value in ("PASS", "TIMEOUT", "MEMORY")
            for value in task_statuses.values()
        )
        and any(value != "PASS" for value in task_statuses.values()),
        "Resource task reports FAIL, ERROR, UNKNOWN or a completed proof.",
    )
    log = validate_public_log(repository, object_fields(receipt.get("log"))).read_text()
    markers = re.findall(r"DONE \((PASS|FAIL|ERROR|UNKNOWN|TIMEOUT), rc=(-?\d+)\)", log)
    require(
        len(markers) <= len(tasks)
        and all(marker in (("PASS", "0"), ("TIMEOUT", "8")) for marker in markers),
        "Resource log reports a proof failure or tool error.",
    )
    failure = re.search(
        r"(?:^|[ \t:])ERROR:|^Traceback |Illegal instruction|Segmentation fault|Broken pipe|"
        r"Assert failed|Status:[ \t]*(?:failed|unknown)|solver returned unknown",
        log,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    require(
        failure is None,
        "Resource native log contains a tool failure or counterexample: "
        + (_log_line(log, failure.start()) if failure is not None else ""),
    )
    native_solver = "yices" if solver == "yices-smt2" else solver
    require(
        re.search(
            r"engine_\d+(?:\.(?:basecase|induction))?:.*\bSolver: "
            + re.escape(native_solver)
            + r"(?:\s|$)",
            log,
        )
        is not None,
        "Resource log does not show the actual admitted solver starting.",
    )
    return {
        "resource_contract": entry["resource_contract"],
        "native_status": status,
        "receipt_sha256": receipt_sha,
        "review_sha256": digest(review_path),
        "wall_seconds": wall,
        "peak_memory_bytes": peak,
    }, receipt


def _row_review(
    repository: Path,
    entry: dict[str, object],
    job: str,
    summaries: list[dict[str, object]],
    receipts: list[dict[str, object]],
    depth_scope: dict[str, object],
) -> Path:
    """Bind the reviewed attempt chain, audit and explicit release-note row."""
    review_path = confined_file(repository, entry.get("review"))
    require(
        digest(review_path) == entry.get("review_sha256"), "Resource row review digest differs."
    )
    review = read_object(review_path)
    require(
        type(review.get("schema_version")) is int and review.get("schema_version") == 1,
        "Unknown resource row review schema.",
    )
    bindings = [
        {
            key: row[key]
            for key in ("resource_contract", "native_status", "receipt_sha256", "review_sha256")
        }
        for row in summaries
    ]
    for name, expected in {
        "verdict": "CLEAR",
        "outcome": "UNPROVEN-RESOURCE",
        "job": job,
        "resource_contract": YICES_RETRY,
        "attempts": bindings,
        "nonvacuity_sha256": entry.get("nonvacuity_sha256"),
        "nonvacuity_review_sha256": entry.get("nonvacuity_review_sha256"),
        "nonvacuity_parent_sha256": entry.get("nonvacuity_parent_sha256"),
        "depth_scope": depth_scope,
        "release_notes": entry.get("release_notes"),
    }.items():
        require(review.get(name) == expected, f"Resource row review {name} binding differs.")
    reviewer = review.get("reviewer")
    require(
        isinstance(reviewer, str)
        and bool(reviewer.strip())
        and all(reviewer != receipt.get("producer") for receipt in receipts),
        "Resource row requires a distinct identified independent reviewer.",
    )
    notes = object_fields(entry.get("release_notes"))
    path = confined_file(repository, notes.get("path"))
    require(digest(path) == notes.get("sha256"), "Resource release notes digest differs.")
    tokens = [job, "UNPROVEN-RESOURCE", str(entry["nonvacuity_sha256"])]
    tokens.extend(str(row["receipt_sha256"]) for row in summaries)
    tokens.extend(
        f"{task}: mode={object_fields(scope)['mode']}; depth={object_fields(scope)['depth']}"
        for task, scope in depth_scope.items()
    )
    require(
        any(all(token in line for token in tokens) for line in path.read_text().splitlines()),
        "Release notes lack the explicit job, outcomes, both receipts, audit and depth scope.",
    )
    return review_path


def validate_resource_row(
    repository: Path,
    job: Path,
    entry: dict[str, object],
    provenance: str,
    solvers: Mapping[str, Path],
    *,
    policy: EvidencePolicy | None = None,
) -> dict[str, object]:
    """Require two actual exhausted solvers and this model's reached enables.

    Parameters
    ----------
    repository : Path
        Current checkout with all immutable technical evidence.
    job : Path
        Current canonical formal configuration.
    entry : dict of str to object
        Explicit UNPROVEN-RESOURCE index entry with reviewed attempts and audit.
    provenance : str
        Verified current SDK provenance.
    solvers : mapping of str to Path
        Actual current solver executables.
    policy : EvidencePolicy or None, optional
        Explicit bounded native qualification policy; absent for production.

    Returns
    -------
    dict of str to object
        Attested UNPROVEN-RESOURCE, aggregate native wall, maximum actual peak,
        all attempt identities and non-vacuity evidence. Never a PASS verdict.

    Raises
    ------
    ValueError
        If any attempt, non-vacuity, reviewed index or release-note binding fails.
    OSError
        If any required technical evidence or native executable is unavailable.
    """
    repository = repository.resolve(strict=True)
    require(
        entry.get("outcome") == "UNPROVEN-RESOURCE"
        and entry.get("resource_contract") == YICES_RETRY,
        "Resource row requires an explicit UNPROVEN-RESOURCE index contract.",
    )
    attempts = entry.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Resource row needs its complete admitted attempt history.")
    summaries, receipts, required_next = validate_resource_history(
        repository, job, attempts, provenance, solvers, policy=policy
    )
    require(
        len(summaries) in (2, 3) and required_next is None,
        "Resource row lacks its final actual Yices attempt.",
    )
    parent_sha = entry.get("nonvacuity_parent_sha256")
    matching = [i for i, row in enumerate(summaries) if row["receipt_sha256"] == parent_sha]
    require(len(matching) == 1, "Resource audit parent is not one exact admitted attempt.")
    parent = matching[0]
    parent_entry = dict(entry)
    parent_attempt = object_fields(attempts[parent])
    if "configuration" in parent_attempt:
        parent_entry["configuration"] = parent_attempt["configuration"]
    else:
        parent_entry.pop("configuration", None)
    policy = EvidencePolicy(REFERENCE_HOST_PROFILE) if policy is None else policy
    inputs = resolve_job(repository, job)
    task_statuses = object_fields(receipts[parent].get("task_statuses"))
    audit = validate_nonvacuity(
        repository,
        inputs,
        parent_entry,
        receipts[parent],
        str(parent_sha),
        list(task_statuses),
        policy=replace(
            policy,
            memory_bytes=policy.audit_memory_bytes,
            timeout_seconds=policy.audit_timeout_seconds,
        ),
    )
    artifact = read_object(confined_file(repository, entry.get("nonvacuity")))
    depth_scope: dict[str, object] = {}
    for task, data in object_fields(artifact.get("tasks")).items():
        options = object_fields(
            object_fields(object_fields(data).get("configuration")).get("options")
        )
        depth_scope[task] = {"mode": options["mode"], "depth": int(str(options["depth"]))}
    review_path = _row_review(
        repository,
        entry,
        inputs.job.relative_to(repository).as_posix(),
        summaries,
        receipts,
        depth_scope,
    )
    return {
        "location": f"{policy.label}, attested",
        "native_status": "UNPROVEN-RESOURCE",
        "resource_contract": YICES_RETRY,
        "attempts": summaries,
        "wall_seconds": sum(float(str(row["wall_seconds"])) for row in summaries),
        "peak_memory_bytes": max(int(str(row["peak_memory_bytes"])) for row in summaries),
        "nonvacuity": audit,
        "depth_scope": depth_scope,
        "review": review_path.relative_to(repository).as_posix(),
    }


def validate_resource_history(
    repository: Path,
    job: Path,
    attempts: object,
    provenance: str,
    solvers: Mapping[str, Path],
    *,
    policy: EvidencePolicy | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    """Check an ordered, once-only chain of actual exhausted native attempts.

    Parameters
    ----------
    repository : Path
        Current resolved checkout.
    job : Path
        Current canonical model configuration.
    attempts : object
        Indexed list of actual resource-attempt receipt and review entries.
    provenance : str
        Current verified SDK provenance.
    solvers : mapping of str to Path
        Actual current solver executables.
    policy : EvidencePolicy or None, optional
        Explicit actual qualification limits, absent for production.

    Returns
    -------
    tuple of list, list and str or None
        Validated summaries, immutable receipt fields and required next contract.

    Raises
    ------
    ValueError
        If history is missing, repeated, reversed, overlapping or out of order,
        or any attempt is a failure rather than an admitted resource stop.
    """
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
        raise ValueError("Resource row needs its complete admitted attempt history.")
    summaries: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    sequence: list[tuple[str, str]] = []
    invocations: set[str] = set()
    previous_stop = 0
    for attempt in attempts:
        summary, receipt = validate_resource_attempt(
            repository, job, object_fields(attempt), provenance, solvers, policy=policy
        )
        measurement = object_fields(receipt.get("measurement"))
        invocation = str(measurement["invocation_id"])
        require(invocation not in invocations, "Resource attempts reuse a native invocation.")
        require(
            int(str(measurement["start_monotonic_microseconds"])) >= previous_stop,
            "Resource attempts overlap or reverse the retained native clock.",
        )
        previous_stop = int(str(measurement["stop_monotonic_microseconds"]))
        invocations.add(invocation)
        summaries.append(summary)
        receipts.append(receipt)
        sequence.append((str(summary["resource_contract"]), str(summary["native_status"])))
        next_attempt(sequence)
    return summaries, receipts, next_attempt(sequence)
