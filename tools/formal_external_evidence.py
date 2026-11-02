# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Attested external formal-result gate

"""Validate source-bound, independently attested external formal receipts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from tools.formal_evidence_validation import (
    EvidencePolicy,
    confined_file,
    digest,
    object_fields,
    read_object,
    require,
    validate_measurement,
)
from tools.formal_execution_accounting import validate_execution_accounting
from tools.formal_public_identity import validate_identity_conversion
from tools.formal_receipt_identity import validate_identity, validate_review
from tools.formal_resource_policy import CHIALVO_REPAIR, HISTORICAL_AMARI, PRIMARY, select_contract
from tools.formal_solver_repair import validate_solver_repair
from tools.formal_resource_evidence import validate_resource_history, validate_resource_row
from tools.formal_log_normalization import validate_public_log
from tools.formal_nonvacuity_evidence import validate_nonvacuity

ML350_POLICY = EvidencePolicy()


def validate_external(
    repository: Path,
    job: Path,
    entry: dict[str, object],
    provenance: str,
    solvers: Mapping[str, Path],
    *,
    policy: EvidencePolicy | None = None,
) -> dict[str, object]:
    """Validate a receipt and its independent review against current inputs.

    Parameters
    ----------
    repository : Path
        Current checkout root.
    job : Path
        Tracked SymbiYosys configuration.
    entry : dict of str to object
        Receipt and review paths with their SHA-256 digests.
    provenance : str
        Verified current SDK provenance.
    solvers : mapping of str to Path
        Actual current solver executables.
    policy : EvidencePolicy, optional
        Production ML350 contract; focused native tests use their actual limits.

    Returns
    -------
    dict of str to object
        Explicit attested location, native verdict and actual measurements.

    Raises
    ------
    ValueError
        If any source, tool, task, log, execution or review binding is invalid.
    OSError
        If required files or native tools are unavailable.
    subprocess.SubprocessError
        If the real SBY parser or solver identification fails.
    """
    repository = repository.resolve(strict=True)
    validate_identity_conversion(repository, entry)
    if entry.get("outcome") == "UNPROVEN-RESOURCE":
        return validate_resource_row(repository, job, entry, provenance, solvers, policy=policy)
    require(entry.get("outcome") in (None, "PASS"), "Unknown external formal outcome.")
    receipt_path = confined_file(repository, entry.get("receipt"))
    receipt_sha = digest(receipt_path)
    require(receipt_sha == entry.get("receipt_sha256"), "Receipt digest differs from the index.")
    receipt = read_object(receipt_path)
    require(
        type(receipt.get("schema_version")) is int and receipt.get("schema_version") == 1,
        "Unknown receipt schema.",
    )
    inputs, task_names, solver_name = validate_identity(
        repository, job, entry, receipt, provenance, solvers
    )
    require(
        receipt.get("task_statuses") == dict.fromkeys(task_names, "PASS"),
        "Every declared SBY task must have an actual PASS.",
    )
    actual_wall, exception_id, registry_digest = validate_execution_accounting(
        repository, receipt, receipt_sha
    )
    log_entry = object_fields(receipt.get("log"))
    log = validate_public_log(repository, log_entry)
    markers = re.findall(r"DONE \((PASS|FAIL|ERROR|UNKNOWN|TIMEOUT), rc=(-?\d+)\)", log.read_text())
    require(
        markers == [("PASS", "0")] * len(task_names),
        "Native final DONE markers disagree with the task verdicts.",
    )
    review_path, review = validate_review(
        repository, entry, receipt, receipt_sha, exception_id, registry_digest
    )
    contract = select_contract(
        entry, review, receipt_sha, inputs.job.relative_to(repository).as_posix()
    )
    require(
        solver_name == contract.solver, "Receipt solver differs from the indexed resource contract."
    )
    qualification_policy = policy
    policy = contract.policy() if policy is None else policy
    wall, peak = validate_measurement(receipt, policy, actual_wall)
    if entry["resource_contract"] == CHIALVO_REPAIR:
        validate_solver_repair(repository, job, entry, receipt, review)
    elif entry["resource_contract"] not in (PRIMARY, HISTORICAL_AMARI):
        history, previous_receipts, expected = validate_resource_history(
            repository,
            job,
            entry.get("predecessors"),
            provenance,
            solvers,
            policy=qualification_policy,
        )
        require(
            expected == entry["resource_contract"],
            "PASS retry lacks its admitted predecessor chain.",
        )
        bindings = [
            {
                key: row[key]
                for key in ("resource_contract", "native_status", "receipt_sha256", "review_sha256")
            }
            for row in history
        ]
        require(
            review.get("predecessors") == bindings,
            "PASS review does not bind the predecessor history.",
        )
        previous = object_fields(previous_receipts[-1].get("measurement"))
        current = object_fields(receipt.get("measurement"))
        require(
            current.get("invocation_id")
            not in [
                object_fields(r.get("measurement")).get("invocation_id") for r in previous_receipts
            ]
            and int(str(current["start_monotonic_microseconds"]))
            >= int(str(previous["stop_monotonic_microseconds"])),
            "PASS retry reuses or predates its predecessor invocation.",
        )
    else:
        require(
            entry.get("predecessors") in (None, []),
            "Primary or historical PASS cannot hide predecessor attempts.",
        )
    return {
        "resource_contract": entry["resource_contract"],
        "location": f"{policy.label}, attested",
        "native_status": "PASS",
        "wall_seconds": wall,
        "peak_memory_bytes": peak,
        "receipt": receipt_path.relative_to(repository).as_posix(),
        "review": review_path.relative_to(repository).as_posix(),
        "nonvacuity": validate_nonvacuity(
            repository,
            inputs,
            entry,
            receipt,
            receipt_sha,
            task_names,
            policy=EvidencePolicy(
                policy.hostname,
                policy.audit_memory_bytes,
                policy.audit_timeout_seconds,
                policy.label,
            ),
        ),
    }


def main(argv: Sequence[str] | None = None, *, policy: EvidencePolicy | None = None) -> int:
    """Identify hosted jobs or validate their indexed external evidence.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; process arguments are used when omitted.
    policy : EvidencePolicy, optional
        Explicit native qualification policy. The CLI always uses production
        limits; local contract tests supply their actual machine and limits.

    Returns
    -------
    int
        Zero for a hosted job or qualified external evidence, two for refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--index", type=Path, default=Path("benchmarks/results/formal/index.json"))
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        repository = args.repository.resolve(strict=True)
        index = read_object(confined_file(repository, args.index.as_posix()))
        require(
            type(index.get("schema_version")) is int and index.get("schema_version") == 1,
            "Unknown external-results index schema.",
        )
        jobs = object_fields(index.get("jobs"))
        job = (repository / args.job).resolve(strict=True)
        require(
            job.is_relative_to(repository) and job.suffix == ".sby" and job.is_file(),
            "Formal job must be a .sby file inside the checkout.",
        )
        name = job.relative_to(repository).as_posix()
        if name not in jobs:
            print(json.dumps({"location": "hosted"}))
            return 0
        solvers = {
            name: Path(executable)
            for name in ("z3", "yices-smt2", "cvc5")
            if (executable := shutil.which(name)) is not None
        }
        print(
            json.dumps(
                validate_external(
                    repository,
                    job,
                    object_fields(jobs[name]),
                    args.provenance.read_text(),
                    solvers,
                    policy=policy,
                )
            )
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"External formal evidence refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
