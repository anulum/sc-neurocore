# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio evidence receipts

"""One receipt shape for every Studio evidence lane.

Simulation runs, analysis results and worker-backed actions each recorded their
own digests in their own words. Nothing could be checked across lanes, and no
lane recorded what its evidence rested on — a digest proves that bytes were not
edited after they were written, not that the analysis beside a simulation was
computed from it.

A receipt carries the cross-runtime seal of its subject (see
:mod:`sc_neurocore.studio.evidence_seal`), the identity scope the subject was
produced under, and the inputs it rests on. An input is named by the value of an
identity field, not by a position in a list, so a receipt means the same thing in
a bundle exported months later as it did in the session that made it. The
identifier is the seal itself, so one artefact exported twice is recognisably one
artefact rather than two.

:mod:`sc_neurocore.studio.evidence_chain` verifies packs of these.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, TypeAlias

from sc_neurocore.studio.evidence_classification import (
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)
from sc_neurocore.studio.evidence_seal import EVIDENCE_SEAL_ALGORITHM, seal_sha256

#: Contract version of the per-subject receipt block.
EVIDENCE_RECEIPT_SCHEMA_VERSION = "studio.evidence-receipt.v1"

#: Key under which a subject payload carries its own receipt.
EVIDENCE_RECEIPT_KEY = "evidence_receipt"

#: How much a receipt attests.
#:
#: ``produced`` — written by the surface that produced the evidence, so it
#: attests what actually ran. ``exported`` — written when a pack was assembled
#: from payloads an operator supplied, so it attests only what the exporter
#: received. A pack must be able to tell the two apart; treating an export-time
#: receipt as proof of provenance is the failure this field exists to prevent.
EvidenceBinding: TypeAlias = Literal["produced", "exported"]

EVIDENCE_BINDINGS: frozenset[str] = frozenset({"exported", "produced"})

UTC = timezone.utc


class EvidenceReceiptError(ValueError):
    """Raised when a receipt is malformed and cannot be read at all."""


@dataclass(frozen=True, slots=True)
class EvidenceDependency:
    """One input a piece of evidence rests on.

    Attributes
    ----------
    lane : str
        Evidence class of the input, such as ``simulation``.
    key : str
        Scope field that identifies the input, such as ``experiment_sha256``.
    value : str
        Value that field must carry on the input.
    """

    lane: StudioEvidenceClassification
    key: str
    value: str

    def to_public_dict(self) -> dict[str, str]:
        """Return the dependency as it is written into a receipt."""
        return {"key": self.key, "lane": self.lane, "value": self.value}


@dataclass(frozen=True, slots=True)
class EvidenceReceipt:
    """What one piece of evidence is, rests on, and was produced under.

    Attributes
    ----------
    receipt_id : str
        Content address ``<lane>.<first 32 characters of the seal>``. It names
        the sealed artefact, so the same artefact exported from any session
        carries the same identifier; the run behind it is named by ``scope``.
    lane : str
        Controlled evidence class.
    status : str
        Terminal status of the action that produced the subject.
    binding : str
        Whether the receipt attests production or only export; see
        :data:`EvidenceBinding`.
    seal_algorithm : str
        Digest algorithm of ``seal_sha256``.
    seal_sha256 : str
        Cross-runtime seal of the subject without its receipt.
    scope : mapping of str to str
        Identity the subject was produced under — model class, descriptor and
        schema digests, numerical profile, experiment digest.
    depends_on : tuple of EvidenceDependency
        Inputs this evidence rests on.
    produced_at_utc : str
        Second-precision UTC timestamp, ``Z``-suffixed.
    """

    receipt_id: str
    lane: StudioEvidenceClassification
    status: StudioEvidenceStatus
    binding: EvidenceBinding
    seal_algorithm: str
    seal_sha256: str
    scope: Mapping[str, str]
    depends_on: tuple[EvidenceDependency, ...]
    produced_at_utc: str

    def to_public_dict(self) -> dict[str, object]:
        """Return the receipt block as it is embedded in a subject payload."""
        return {
            "binding": self.binding,
            "depends_on": [dependency.to_public_dict() for dependency in self.depends_on],
            "lane": self.lane,
            "produced_at_utc": self.produced_at_utc,
            "receipt_id": self.receipt_id,
            "schema_version": EVIDENCE_RECEIPT_SCHEMA_VERSION,
            "scope": dict(sorted(self.scope.items())),
            "seal_algorithm": self.seal_algorithm,
            "seal_sha256": self.seal_sha256,
            "status": self.status,
        }


def subject_of(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the sealed part of a payload: everything but its own receipt.

    Parameters
    ----------
    payload : mapping
        A payload that may already carry a receipt.

    Returns
    -------
    dict
        The payload without :data:`EVIDENCE_RECEIPT_KEY`.
    """
    return {key: value for key, value in payload.items() if key != EVIDENCE_RECEIPT_KEY}


def build_evidence_receipt(
    payload: Mapping[str, object],
    *,
    lane: str,
    status: str,
    binding: str,
    scope: Mapping[str, str],
    depends_on: Sequence[EvidenceDependency] = (),
    produced_at_utc: str,
) -> EvidenceReceipt:
    """Seal a payload and describe what it rests on.

    Parameters
    ----------
    payload : mapping
        The evidence payload. Any receipt already present is excluded from the
        seal, so re-sealing an exported payload reproduces the same receipt.
    lane : str
        Controlled evidence class.
    status : str
        Terminal status of the producing action.
    binding : str
        ``produced`` or ``exported``; see :data:`EvidenceBinding`.
    scope : mapping of str to str
        Identity fields the subject was produced under.
    depends_on : sequence of EvidenceDependency
        Inputs this evidence rests on.
    produced_at_utc : str
        Second-precision UTC timestamp, ``Z``-suffixed.

    Returns
    -------
    EvidenceReceipt
        The receipt for this payload.

    Raises
    ------
    EvidenceReceiptError
        If the lane, status or timestamp is not valid.
    EvidenceSealError
        If the payload cannot be sealed identically in both runtimes.
    """
    validated_lane = validate_lane(lane)
    validated_status = validate_status(status)
    validated_binding = validate_binding(binding)
    parse_timestamp(produced_at_utc)
    seal = seal_sha256(subject_of(payload))
    return EvidenceReceipt(
        receipt_id=f"{validated_lane}.{seal[:32]}",
        lane=validated_lane,
        status=validated_status,
        binding=validated_binding,
        seal_algorithm=EVIDENCE_SEAL_ALGORITHM,
        seal_sha256=seal,
        scope={str(key): str(value) for key, value in scope.items()},
        depends_on=tuple(depends_on),
        produced_at_utc=produced_at_utc,
    )


def attach_evidence_receipt(
    payload: Mapping[str, object],
    *,
    lane: str,
    status: str,
    binding: str,
    scope: Mapping[str, str],
    depends_on: Sequence[EvidenceDependency] = (),
    now: datetime | None = None,
) -> dict[str, object]:
    """Return the payload with its receipt embedded.

    Parameters
    ----------
    payload : mapping
        The evidence payload.
    lane : str
        Controlled evidence class.
    status : str
        Terminal status of the producing action.
    binding : str
        ``produced`` or ``exported``; see :data:`EvidenceBinding`.
    scope : mapping of str to str
        Identity fields the subject was produced under.
    depends_on : sequence of EvidenceDependency
        Inputs this evidence rests on.
    now : datetime, optional
        Production time; the current UTC time when omitted.

    Returns
    -------
    dict
        A new payload carrying :data:`EVIDENCE_RECEIPT_KEY`.
    """
    receipt = build_evidence_receipt(
        payload,
        lane=lane,
        status=status,
        binding=binding,
        scope=scope,
        depends_on=depends_on,
        produced_at_utc=utc_timestamp(now),
    )
    result = subject_of(payload)
    result[EVIDENCE_RECEIPT_KEY] = receipt.to_public_dict()
    return result


def read_evidence_receipt(payload: Mapping[str, object]) -> EvidenceReceipt | None:
    """Return the receipt embedded in a payload, or ``None`` when absent.

    Parameters
    ----------
    payload : mapping
        A payload that may carry a receipt.

    Returns
    -------
    EvidenceReceipt or None
        The parsed receipt, or ``None`` for a payload written before receipts
        existed.

    Raises
    ------
    EvidenceReceiptError
        If a receipt is present but malformed. A receipt that cannot be read is
        never treated as an absent one.
    """
    block = payload.get(EVIDENCE_RECEIPT_KEY)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise EvidenceReceiptError("An evidence receipt must be an object.")
    if block.get("schema_version") != EVIDENCE_RECEIPT_SCHEMA_VERSION:
        raise EvidenceReceiptError(
            f"Evidence receipt schema {block.get('schema_version')!r} is not the "
            f"contract this build reads ({EVIDENCE_RECEIPT_SCHEMA_VERSION})."
        )
    seal = block.get("seal_sha256")
    if not is_sha256_hex(seal):
        raise EvidenceReceiptError("An evidence receipt requires a SHA-256 seal.")
    if block.get("seal_algorithm") != EVIDENCE_SEAL_ALGORITHM:
        raise EvidenceReceiptError("An evidence receipt must name the sha256 seal algorithm.")
    receipt_id = block.get("receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id:
        raise EvidenceReceiptError("An evidence receipt requires an identifier.")
    produced_at_utc = block.get("produced_at_utc")
    if not isinstance(produced_at_utc, str):
        raise EvidenceReceiptError("An evidence receipt requires a production timestamp.")
    parse_timestamp(produced_at_utc)
    return EvidenceReceipt(
        receipt_id=receipt_id,
        lane=validate_lane(block.get("lane")),
        status=validate_status(block.get("status")),
        binding=validate_binding(block.get("binding")),
        seal_algorithm=EVIDENCE_SEAL_ALGORITHM,
        seal_sha256=str(seal),
        scope=_scope(block.get("scope")),
        depends_on=_dependencies(block.get("depends_on")),
        produced_at_utc=produced_at_utc,
    )


def utc_timestamp(now: datetime | None = None) -> str:
    """Return a second-precision ``Z``-suffixed UTC timestamp.

    Parameters
    ----------
    now : datetime, optional
        Moment to render; the current UTC time when omitted.

    Returns
    -------
    str
        For example ``2026-09-06T11:22:33Z``.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    return moment.isoformat().replace("+00:00", "Z")


def validate_lane(value: object) -> StudioEvidenceClassification:
    """Return a controlled evidence class or refuse the receipt.

    Parameters
    ----------
    value : object
        Candidate lane read from a receipt.

    Returns
    -------
    StudioEvidenceClassification
        The validated evidence class.

    Raises
    ------
    EvidenceReceiptError
        If the value is not a Studio evidence class.
    """
    if not isinstance(value, str):
        raise EvidenceReceiptError("An evidence receipt requires an evidence class.")
    try:
        return validate_studio_evidence_classification(value)
    except ValueError as exc:
        raise EvidenceReceiptError(f"{value!r} is not a Studio evidence class.") from exc


def validate_status(value: object) -> StudioEvidenceStatus:
    """Return a controlled terminal status or refuse the receipt.

    Parameters
    ----------
    value : object
        Candidate status read from a receipt.

    Returns
    -------
    StudioEvidenceStatus
        The validated terminal status.

    Raises
    ------
    EvidenceReceiptError
        If the value is not a terminal evidence status.
    """
    if not isinstance(value, str):
        raise EvidenceReceiptError("An evidence receipt requires a terminal status.")
    try:
        return validate_studio_evidence_status(value)
    except ValueError as exc:
        raise EvidenceReceiptError(f"{value!r} is not a terminal evidence status.") from exc


def validate_binding(value: object) -> EvidenceBinding:
    """Return a controlled receipt binding or refuse the receipt.

    Parameters
    ----------
    value : object
        Candidate binding read from a receipt.

    Returns
    -------
    EvidenceBinding
        ``produced`` or ``exported``.

    Raises
    ------
    EvidenceReceiptError
        If the value names neither.
    """
    if value == "produced":
        return "produced"
    if value == "exported":
        return "exported"
    raise EvidenceReceiptError(
        f"{value!r} is not an evidence receipt binding; expected one of "
        f"{', '.join(sorted(EVIDENCE_BINDINGS))}."
    )


def parse_timestamp(value: str) -> datetime:
    """Return the moment a ``Z``-suffixed UTC timestamp names.

    Parameters
    ----------
    value : str
        Timestamp text from a receipt.

    Returns
    -------
    datetime
        The parsed, timezone-aware moment.

    Raises
    ------
    EvidenceReceiptError
        If the text is not a UTC ISO-8601 timestamp.
    """
    if not value.endswith("Z"):
        raise EvidenceReceiptError("An evidence timestamp must be UTC and end with Z.")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceReceiptError(f"{value!r} is not an ISO-8601 timestamp.") from exc


def is_sha256_hex(value: object) -> bool:
    """Return whether ``value`` is a lowercase 64-character SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scope(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise EvidenceReceiptError("An evidence receipt scope must be an object.")
    scope: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise EvidenceReceiptError("An evidence receipt scope must map names to text.")
        scope[key] = item
    return scope


def _dependencies(value: object) -> tuple[EvidenceDependency, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvidenceReceiptError("Evidence receipt dependencies must be a list.")
    dependencies: list[EvidenceDependency] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise EvidenceReceiptError("Each evidence dependency must be an object.")
        key = item.get("key")
        reference = item.get("value")
        if not isinstance(key, str) or not key or not isinstance(reference, str) or not reference:
            raise EvidenceReceiptError("Each evidence dependency requires a key and a value.")
        dependencies.append(
            EvidenceDependency(lane=validate_lane(item.get("lane")), key=key, value=reference)
        )
    return tuple(dependencies)


__all__ = [
    "EVIDENCE_BINDINGS",
    "EVIDENCE_RECEIPT_KEY",
    "EVIDENCE_RECEIPT_SCHEMA_VERSION",
    "EvidenceBinding",
    "EvidenceDependency",
    "EvidenceReceipt",
    "EvidenceReceiptError",
    "attach_evidence_receipt",
    "build_evidence_receipt",
    "is_sha256_hex",
    "parse_timestamp",
    "read_evidence_receipt",
    "subject_of",
    "utc_timestamp",
    "validate_binding",
    "validate_lane",
    "validate_status",
]
