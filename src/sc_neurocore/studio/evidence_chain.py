# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio evidence chain verification

"""Check a pack of evidence against its own receipts.

Every exported artefact carried a digest of itself, which proves only that its
bytes were not edited after it was written. It did not say whether the analysis
in the bundle was computed from the simulation beside it, whether the two agree
about which model ran, or whether either is still current.

:func:`verify_evidence_chain` re-seals every subject, resolves each declared
input inside the pack, and reaches one verdict per subject:

``verified``
    The payload matches its receipt and rests on inputs that do too.
``tampered``
    The payload no longer seals to the digest its receipt records.
``missing_dependency``
    An input the receipt names is not in this pack. The link cannot be checked
    here; a pack that carries a run without its inputs is incomplete, not
    self-contradictory, so this does not refuse an export.
``scope_mismatch``
    The subject and a resolved input disagree about an identity field — the
    wrong model or the wrong numerical profile.
``stale``
    The subject predates the input it claims, or descends from something that
    failed a check. Descending from an input the pack merely could not resolve
    is not staleness: nothing about the subject was found wrong.
``unsealed``
    The payload comes from a build that wrote no receipt. It is reported, never
    silently passed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias

from sc_neurocore.studio.evidence_receipt import (
    EvidenceReceipt,
    EvidenceReceiptError,
    parse_timestamp,
    read_evidence_receipt,
    subject_of,
    utc_timestamp,
)
from sc_neurocore.studio.evidence_seal import (
    EVIDENCE_SEAL_ALGORITHM,
    EvidenceSealError,
    seal_sha256,
)

#: Contract version of the chain document written beside a bundle manifest.
EVIDENCE_CHAIN_SCHEMA_VERSION = "studio.evidence-chain.v1"

#: Version of the verifying implementation. A verdict is only meaningful with
#: the verifier that produced it, so every report names this.
EVIDENCE_VERIFIER_VERSION = "studio.evidence-verifier.v1"

#: Verdicts that mean the pack contradicts itself and must not be published as
#: verified. ``unsealed`` and ``missing_dependency`` are deliberately absent:
#: each says the verifier could not check something, which is a boundary to
#: report, not a check that failed.
CONTRADICTING_VERDICTS: frozenset[str] = frozenset({"scope_mismatch", "stale", "tampered"})

EvidenceVerdict: TypeAlias = Literal[
    "verified",
    "tampered",
    "missing_dependency",
    "scope_mismatch",
    "stale",
    "unsealed",
]

_ScopeIndex: TypeAlias = Mapping[tuple[str, str, str], tuple[str, EvidenceReceipt]]


@dataclass(frozen=True, slots=True)
class EvidenceChainEntry:
    """The verdict reached for one subject in a pack.

    Attributes
    ----------
    name : str
        Pack-relative name of the subject, used for operator reporting.
    verdict : str
        One of the :data:`EvidenceVerdict` values.
    receipt_id : str
        Receipt identifier, empty for an ``unsealed`` subject.
    reason : str
        What the verifier observed, in words an operator can act on.
    """

    name: str
    verdict: EvidenceVerdict
    receipt_id: str
    reason: str

    def to_public_dict(self) -> dict[str, str]:
        """Return the entry as it is written into a chain document."""
        return {
            "name": self.name,
            "reason": self.reason,
            "receipt_id": self.receipt_id,
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class EvidenceChainReport:
    """The result of verifying a whole pack.

    Attributes
    ----------
    entries : tuple of EvidenceChainEntry
        One entry per subject, in the order supplied.
    verified : bool
        True when nothing in the pack contradicts anything else in it.
    complete : bool
        True when every subject was checkable and checked. A pack can be
        verified without being complete: exporting a run without its inputs
        leaves links this pack cannot check.
    verified_at_utc : str
        When the verification ran.
    """

    entries: tuple[EvidenceChainEntry, ...]
    verified: bool
    complete: bool
    verified_at_utc: str

    def verdict_counts(self) -> dict[str, int]:
        """Return how many subjects reached each verdict."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.verdict] = counts.get(entry.verdict, 0) + 1
        return dict(sorted(counts.items()))

    def unverified(self) -> tuple[EvidenceChainEntry, ...]:
        """Return every entry that did not verify, including unsealed ones."""
        return tuple(entry for entry in self.entries if entry.verdict != "verified")

    def contradicted(self) -> tuple[EvidenceChainEntry, ...]:
        """Return every entry whose verdict contradicts the rest of the pack."""
        return tuple(entry for entry in self.entries if entry.verdict in CONTRADICTING_VERDICTS)

    def to_public_dict(self) -> dict[str, object]:
        """Return the chain document written beside a bundle manifest."""
        return {
            "complete": self.complete,
            "entries": [entry.to_public_dict() for entry in self.entries],
            "schema_version": EVIDENCE_CHAIN_SCHEMA_VERSION,
            "seal_algorithm": EVIDENCE_SEAL_ALGORITHM,
            "verdict_counts": self.verdict_counts(),
            "verified": self.verified,
            "verified_at_utc": self.verified_at_utc,
            "verifier_version": EVIDENCE_VERIFIER_VERSION,
        }


def verify_evidence_chain(
    subjects: Mapping[str, Mapping[str, object]],
    *,
    now: datetime | None = None,
) -> EvidenceChainReport:
    """Verify a whole pack of evidence against its own receipts.

    Parameters
    ----------
    subjects : mapping of str to mapping
        Pack-relative name to payload, as read back from the exported files.
    now : datetime, optional
        Verification time recorded in the report.

    Returns
    -------
    EvidenceChainReport
        One verdict per subject, plus whether the pack verified as a whole.

    Raises
    ------
    EvidenceReceiptError
        If a receipt is present but cannot be read. A malformed receipt is a
        broken pack, not a subject that merely failed.
    """
    receipts: dict[str, EvidenceReceipt | None] = {}
    resealed: dict[str, str] = {}
    for name, payload in subjects.items():
        receipt = read_evidence_receipt(payload)
        receipts[name] = receipt
        if receipt is None:
            continue
        try:
            resealed[name] = seal_sha256(subject_of(payload))
        except EvidenceSealError:
            resealed[name] = ""

    index = _scope_index(receipts)
    verdicts: dict[str, tuple[EvidenceVerdict, str]] = {}
    inputs: dict[str, tuple[str, ...]] = {}
    for name, receipt in receipts.items():
        verdict, reason, resolved = _direct_verdict(receipt, resealed.get(name, ""), index)
        verdicts[name] = (verdict, reason)
        inputs[name] = resolved

    _propagate_stale(verdicts, inputs)
    entries = tuple(
        EvidenceChainEntry(
            name=name,
            verdict=verdicts[name][0],
            receipt_id=_receipt_id(receipts[name]),
            reason=verdicts[name][1],
        )
        for name in subjects
    )
    return EvidenceChainReport(
        entries=entries,
        verified=not any(entry.verdict in CONTRADICTING_VERDICTS for entry in entries),
        complete=all(entry.verdict == "verified" for entry in entries),
        verified_at_utc=utc_timestamp(now),
    )


def _receipt_id(receipt: EvidenceReceipt | None) -> str:
    return receipt.receipt_id if receipt is not None else ""


def _direct_verdict(
    receipt: EvidenceReceipt | None,
    resealed: str,
    index: _ScopeIndex,
) -> tuple[EvidenceVerdict, str, tuple[str, ...]]:
    """Return the verdict a subject reaches on its own evidence."""
    if receipt is None:
        return ("unsealed", "The payload carries no evidence receipt.", ())
    if resealed != receipt.seal_sha256:
        return (
            "tampered",
            f"The payload re-seals to {resealed or 'nothing sealable'}, "
            f"not to the recorded {receipt.seal_sha256}.",
            (),
        )
    resolved: list[str] = []
    for dependency in receipt.depends_on:
        found = index.get((dependency.lane, dependency.key, dependency.value))
        if found is None:
            return (
                "missing_dependency",
                f"No {dependency.lane} evidence in the pack carries "
                f"{dependency.key}={dependency.value}.",
                (),
            )
        name, parent = found
        conflict = _scope_conflict(receipt.scope, parent.scope)
        if conflict is not None:
            field, mine, theirs = conflict
            return (
                "scope_mismatch",
                f"It records {field}={mine} but its input {name} records {theirs}.",
                (),
            )
        if parse_timestamp(receipt.produced_at_utc) < parse_timestamp(parent.produced_at_utc):
            return (
                "stale",
                f"It was produced at {receipt.produced_at_utc}, before its input "
                f"{name} at {parent.produced_at_utc}.",
                (),
            )
        resolved.append(name)
    return ("verified", "The payload matches its receipt.", tuple(resolved))


def _scope_conflict(
    child: Mapping[str, str], parent: Mapping[str, str]
) -> tuple[str, str, str] | None:
    """Return the first identity field on which two receipts disagree.

    A field absent from either side is not a disagreement: lanes record
    different amounts of identity, and silence is not a contradiction.
    """
    for field in sorted(set(child) & set(parent)):
        if child[field] != parent[field]:
            return (field, child[field], parent[field])
    return None


def _scope_index(
    receipts: Mapping[str, EvidenceReceipt | None],
) -> dict[tuple[str, str, str], tuple[str, EvidenceReceipt]]:
    """Map every ``(lane, scope key, value)`` in the pack to the subject holding it.

    The first subject to declare a value keeps it, so a pack carrying the same
    run twice resolves dependencies to one of them rather than to neither.
    """
    index: dict[tuple[str, str, str], tuple[str, EvidenceReceipt]] = {}
    for name, receipt in receipts.items():
        if receipt is None:
            continue
        for key, value in receipt.scope.items():
            index.setdefault((receipt.lane, key, value), (name, receipt))
    return index


def _propagate_stale(
    verdicts: dict[str, tuple[EvidenceVerdict, str]],
    inputs: Mapping[str, tuple[str, ...]],
) -> None:
    """Downgrade every subject that descends from a failed check.

    Only a contradicting verdict travels: a subject whose input the pack could
    not resolve is not thereby wrong, and calling it stale would turn an
    incomplete export into a refused one. Repeated until nothing changes, so a
    verdict reaches the whole length of a chain rather than one edge of it.
    """
    changed = True
    while changed:
        changed = False
        for name, resolved in inputs.items():
            if verdicts[name][0] != "verified":
                continue
            for parent in resolved:
                if verdicts[parent][0] in CONTRADICTING_VERDICTS:
                    verdicts[name] = (
                        "stale",
                        f"Its input {parent} did not verify ({verdicts[parent][0]}).",
                    )
                    changed = True
                    break


__all__ = [
    "CONTRADICTING_VERDICTS",
    "EVIDENCE_CHAIN_SCHEMA_VERSION",
    "EVIDENCE_VERIFIER_VERSION",
    "EvidenceChainEntry",
    "EvidenceChainReport",
    "EvidenceReceiptError",
    "EvidenceVerdict",
    "verify_evidence_chain",
]
