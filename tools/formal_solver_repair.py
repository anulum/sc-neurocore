# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact Chialvo solver-crash repair admission

"""Keep one reviewed solver repair separate from ordinary resource retries."""

from __future__ import annotations

from pathlib import Path

from tools.formal_evidence_validation import (
    confined_file,
    digest,
    object_fields,
    read_object,
    require,
)
from tools.formal_job_inputs import resolve_job
from tools.formal_log_normalization import validate_public_log
from tools.formal_receipt_identity import validate_review
from tools.formal_resource_policy import CHIALVO_JOB, CHIALVO_REPAIR

ERROR_RECEIPT = "4967f867e68be34bdfdac077dff55f9dd63133391c55072ac0a67fc6de16d48f"
ERROR_DIAGNOSTIC = "f29b2cac3fac2aa00bfedc73f0e8f96f85391b240343925e1339fa22d2bfd22f"
REPAIRED_CONFIGURATION = "46f40e22ff28f6538d14872f09b4ecddcbdc3ddba86f4171f2ef1a59acdba7be"
PORTABLE_YICES = "524357fd625311f9a43affded6ea4250c19783f07b2119d82d57ebb25c75d50e"


def validate_solver_repair(
    repository: Path,
    job: Path,
    entry: dict[str, object],
    receipt: dict[str, object],
    review: dict[str, object],
) -> None:
    """Require the exact reviewed ERROR parent and a distinct successful repair.

    Parameters
    ----------
    repository : Path
        Resolved checkout containing the original error and new proof receipts.
    job : Path
        Current canonical Chialvo SBY configuration.
    entry : dict of str to object
        Indexed repair contract, configuration and original error bindings.
    receipt : dict of str to object
        New native PASS receipt, checked by the external evidence validator.
    review : dict of str to object
        Independent review of the new proof and its exact repair lineage.

    Raises
    ------
    ValueError
        If another job, error, diagnosis, solver, source or invocation is used.
    OSError
        If retained error evidence or current sources are unavailable.
    """
    inputs = resolve_job(repository, job)
    require(
        inputs.job.relative_to(repository).as_posix() == CHIALVO_JOB
        and entry.get("resource_contract") == CHIALVO_REPAIR,
        "Solver repair is limited to the explicit Chialvo contract.",
    )
    require(entry.get("predecessors") in (None, []), "Solver repair cannot use resource history.")
    parent_entry = object_fields(entry.get("solver_error"))
    path = confined_file(repository, parent_entry.get("receipt"))
    require(
        digest(path) == parent_entry.get("receipt_sha256") == ERROR_RECEIPT,
        "Solver repair requires the exact immutable Chialvo ERROR receipt.",
    )
    parent = read_object(path)
    require(
        parent.get("source_hashes") == inputs.source_hashes(), "Original ERROR sources are stale."
    )
    _, parent_review = validate_review(repository, parent_entry, parent, ERROR_RECEIPT)
    require(
        parent_review.get("native_status") == "ERROR"
        and parent_review.get("classification") == "SOLVER-CRASH",
        "Independent parent review does not attest the solver crash.",
    )
    diagnosis = confined_file(repository, parent_entry.get("diagnostic"))
    require(
        digest(diagnosis) == parent_entry.get("diagnostic_sha256") == ERROR_DIAGNOSTIC
        and parent_review.get("diagnostic_sha256") == ERROR_DIAGNOSTIC,
        "Solver repair requires the exact reviewed crash diagnosis.",
    )
    validate_public_log(repository, parent.get("log"))
    configuration = object_fields(entry.get("configuration"))
    require(
        digest(confined_file(repository, configuration.get("path")))
        == configuration.get("sha256")
        == REPAIRED_CONFIGURATION,
        "Solver repair configuration differs from the reviewed variant.",
    )
    solver = object_fields(object_fields(receipt.get("toolchain")).get("solver"))
    require(
        solver.get("name") == "yices-smt2" and solver.get("sha256") == PORTABLE_YICES,
        "Solver repair requires the reviewed portable Yices binary.",
    )
    binding = {
        "resource_contract": CHIALVO_REPAIR,
        "error_receipt_sha256": ERROR_RECEIPT,
        "error_review_sha256": parent_entry.get("review_sha256"),
        "diagnostic_sha256": ERROR_DIAGNOSTIC,
        "configuration_sha256": REPAIRED_CONFIGURATION,
    }
    require(
        review.get("verdict") == "CLEAR" and review.get("solver_repair") == binding,
        "PASS review does not bind the exact solver repair lineage.",
    )
    previous = object_fields(parent.get("measurement"))
    current = object_fields(receipt.get("measurement"))
    require(
        current.get("status") == "PASS"
        and current.get("invocation_id") != previous.get("invocation_id")
        and type(current.get("start_monotonic_microseconds")) is int
        and int(str(current["start_monotonic_microseconds"]))
        >= int(str(previous["stop_monotonic_microseconds"])),
        "Solver repair is not a distinct subsequent native PASS.",
    )
