# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit formal resource contracts and retry order

"""Select reviewed resource contracts without inferring limits from measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from tools.formal_evidence_validation import EvidencePolicy, require

PRIMARY = "z3-primary-40-v1"
MEMORY_RETRY = "z3-memory-retry-64-v1"
YICES_RETRY = "yices-resource-retry-64-v1"
HISTORICAL_AMARI = "amari-historical-64-v1"
CHIALVO_REPAIR = "chialvo-solver-repair-yices-64-v1"
CHIALVO_JOB = "hdl/formal/catalogue/sc_chialvo_map.sby"
AMARI_RECEIPT = "188b5cd2741c55b2bcc4b3331971c263a9c30a05b9b1948a0f48ab463471bd75"
AMARI_JOB = "hdl/formal/catalogue/sc_amari_field.sby"


@dataclass(frozen=True)
class ResourceContract:
    """Bind an explicit contract identifier to a solver and fixed limits.

    Parameters
    ----------
    solver : str
        Required current SDK solver executable name.
    memory_bytes : int
        Admitted cgroup cap, independent of the observed memory peak.
    timeout_seconds : int
        Admitted primary-service runtime ceiling.
    """

    solver: str
    memory_bytes: int
    timeout_seconds: int = 7200

    def policy(self) -> EvidencePolicy:
        """Return the production policy for this registered contract.

        Returns
        -------
        EvidencePolicy
            ML350 host and audit limits with this fixed primary budget.
        """
        return replace(
            EvidencePolicy(),
            memory_bytes=self.memory_bytes,
            timeout_seconds=self.timeout_seconds,
        )


CONTRACTS: Mapping[str, ResourceContract] = {
    PRIMARY: ResourceContract("z3", 40 * 1024**3),
    MEMORY_RETRY: ResourceContract("z3", 64 * 1024**3),
    YICES_RETRY: ResourceContract("yices-smt2", 64 * 1024**3),
    HISTORICAL_AMARI: ResourceContract("z3", 64 * 1024**3),
    CHIALVO_REPAIR: ResourceContract("yices-smt2", 64 * 1024**3),
}


def select_contract(
    entry: Mapping[str, object],
    review: Mapping[str, object],
    receipt_sha256: str,
    job: str,
) -> ResourceContract:
    """Require an indexed contract and the same independently reviewed choice.

    Parameters
    ----------
    entry : mapping of str to object
        Trusted, digest-bound index entry. Receipt memory never selects it.
    review : mapping of str to object
        Actual independently reviewed record, checked by the receipt validator.
    receipt_sha256 : str
        Digest of the actual receipt bound by both index and review.
    job : str
        Current repository-relative configuration path.

    Returns
    -------
    ResourceContract
        Registered immutable solver and production resource limits.

    Raises
    ------
    ValueError
        If the contract is absent, unknown, unreviewed or the historical
        exception is applied to any other receipt or job.
    """
    name = entry.get("resource_contract")
    require(isinstance(name, str) and name in CONTRACTS, "Missing or unknown resource contract.")
    require(
        review.get("verdict") == "CLEAR"
        and review.get("receipt_sha256") == receipt_sha256
        and review.get("resource_contract") == name,
        "Independent review does not bind the indexed resource contract.",
    )
    if name == CHIALVO_REPAIR:
        require(job == CHIALVO_JOB, "Solver repair applies only to Chialvo.")
    if name == HISTORICAL_AMARI:
        require(
            receipt_sha256 == AMARI_RECEIPT and job == AMARI_JOB,
            "Historical 64 GiB contract applies only to the exact Amari receipt.",
        )
    return CONTRACTS[str(name)]


def next_attempt(attempts: Sequence[tuple[str, str]]) -> str | None:
    """Check the fixed attempt order and identify the next admitted contract.

    Parameters
    ----------
    attempts : sequence of tuple of str and str
        Already validated contract identifiers and actual native outcomes,
        in execution order. This scheduling decision is not a proof verdict.

    Returns
    -------
    str or None
        Required next contract, or no further attempt after PASS or both
        solvers exhausting their admitted resources. The latter still needs
        model-specific non-vacuity, independent review and an indexed
        UNPROVEN-RESOURCE entry; it never establishes PASS.

    Raises
    ------
    ValueError
        If an attempt is repeated, out of order, follows a terminal result,
        has an unrecognised outcome or reports FAIL, ERROR or UNKNOWN.
    """
    expected: str | None = PRIMARY
    for contract, outcome in attempts:
        require(expected is not None and contract == expected, "Formal attempt order differs.")
        require(
            outcome in ("PASS", "TIMEOUT", "MEMORY"),
            "Formal FAIL, ERROR, UNKNOWN or invalid outcome remains RED.",
        )
        if outcome == "PASS" or contract == YICES_RETRY:
            expected = None
        elif contract == PRIMARY and outcome == "MEMORY":
            expected = MEMORY_RETRY
        else:
            expected = YICES_RETRY
    return expected
