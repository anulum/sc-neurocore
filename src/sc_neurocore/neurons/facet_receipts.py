# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Immutable facet receipts and the evidence dependency matrix

"""Immutable facet receipts: what a credited readiness facet must carry.

A descriptor *declares* readiness (``validation.dynamics_faithful``,
``silicon.cosim_validated``, …). A facet receipt records that the declared
evidence command was actually executed: which subjects it covered (by content
digest), which command ran under which tool and runtime, and its outcome.
The readiness verifier (:mod:`sc_neurocore.neurons.readiness`) credits a
facet only from a receipt that

* is sealed (its ``receipt_sha256`` matches its content),
* names the facet and the class it is being read for,
* carries every required subject kind for that facet with a digest that still
  matches the current repository content,
* ended with ``outcome = "passed"``, exit code 0, at least one passed check and
  no failed, errored or skipped check, and
* carries the claim scope the facet requires (a bounded safety proof is
  recorded under ``formal_safety`` and can never credit ``formal_equivalence``).

Receipts are append-only: a later run writes a new file; the verifier reads the
newest receipt per (class, facet) and never edits an older one.

The dependency matrix :data:`INVALIDATION_MATRIX` names, per facet, the subject
kinds whose content change invalidates the receipt. Changing one subject
therefore invalidates exactly the facets that list its kind and leaves every
other receipt valid.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from sc_neurocore.neurons.evidence_references import sha256_canonical_json

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

FACET_RECEIPT_SCHEMA = "sc-neurocore.facet-receipt.v1"
RECEIPT_DIR = Path(__file__).resolve().parent / "facet_receipts"

Axis = Literal["science", "software", "silicon"]
SubjectKind = Literal[
    "descriptor-contract",
    "model-module",
    "schema-profile",
    "source-reference",
    "compiler",
    "committed-rtl",
    "native-backend",
    "report",
    "validator",
]
SubjectScope = Literal["file", "contract-sections", "tree"]
Outcome = Literal["passed", "failed", "error", "skipped", "timeout"]

SUBJECT_KINDS: tuple[SubjectKind, ...] = (
    "descriptor-contract",
    "model-module",
    "schema-profile",
    "source-reference",
    "compiler",
    "committed-rtl",
    "native-backend",
    "report",
    "validator",
)
OUTCOMES: frozenset[str] = frozenset({"passed", "failed", "error", "skipped", "timeout"})
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class FacetReceiptError(ValueError):
    """Raised when a receipt payload violates the receipt contract."""


@dataclass(frozen=True, slots=True)
class FacetSpec:
    """Definition of one readiness facet.

    Parameters
    ----------
    name:
        Facet identifier (``backend:rust``, ``cosim``, …).
    axis:
        ``science`` (S4/S5), ``software`` (per-backend completion) or
        ``silicon`` (H0–H5 and the physical rung beyond them).
    rung:
        Tier rung the facet credits on its axis, or ``None`` when it credits no
        tier (per-backend completion, bounded safety, physical measurement).
    required_subjects:
        Subject kinds a creditable receipt must carry.
    optional_subjects:
        Subject kinds a receipt may carry; they are checked for freshness when
        present.
    evidence_field:
        Descriptor field holding the declared evidence, ``""`` when the
        descriptor has no field for the facet.
    claim_scope:
        Claim scope a receipt must state, ``""`` when unconstrained.
    """

    name: str
    axis: Axis
    rung: int | None
    required_subjects: tuple[SubjectKind, ...]
    optional_subjects: tuple[SubjectKind, ...] = ()
    evidence_field: str = ""
    claim_scope: str = ""

    @property
    def subjects(self) -> tuple[SubjectKind, ...]:
        """Every subject kind whose change invalidates the facet."""
        return self.required_subjects + self.optional_subjects


_SCIENCE_SUBJECTS: tuple[SubjectKind, ...] = ("descriptor-contract", "model-module", "validator")
_SCIENCE_OPTIONAL: tuple[SubjectKind, ...] = ("schema-profile", "source-reference")
_HARDWARE_REPORT: tuple[SubjectKind, ...] = ("committed-rtl", "report", "validator")

FACETS: tuple[FacetSpec, ...] = (
    FacetSpec(
        "dynamics_faithful",
        "science",
        4,
        _SCIENCE_SUBJECTS,
        _SCIENCE_OPTIONAL,
        "validation.evidence",
    ),
    FacetSpec(
        "class_validated",
        "science",
        5,
        _SCIENCE_SUBJECTS,
        _SCIENCE_OPTIONAL,
        "validation.evidence",
    ),
    *(
        FacetSpec(
            f"backend:{backend}",
            "software",
            None,
            ("descriptor-contract", "model-module", "validator"),
            ("native-backend", "schema-profile", "source-reference"),
        )
        for backend in ("python", "rust", "julia", "go", "mojo")
    ),
    FacetSpec(
        "rtl_compile",
        "silicon",
        0,
        ("schema-profile", "compiler", "validator"),
        ("committed-rtl",),
        "silicon.cosim_evidence",
    ),
    FacetSpec(
        "cosim",
        "silicon",
        1,
        ("descriptor-contract", "model-module", "schema-profile", "compiler", "validator"),
        ("committed-rtl", "source-reference"),
        "silicon.cosim_evidence",
        "trace",
    ),
    FacetSpec(
        "synthesis",
        "silicon",
        2,
        ("committed-rtl", "report", "validator"),
        ("schema-profile", "compiler"),
        "silicon.synth_report",
    ),
    FacetSpec("timing", "silicon", 3, _HARDWARE_REPORT, (), "silicon.timing_report"),
    FacetSpec(
        "formal_equivalence",
        "silicon",
        4,
        ("committed-rtl", "compiler", "report", "validator"),
        ("schema-profile",),
        "silicon.equivalence_proof",
        "equivalence",
    ),
    FacetSpec(
        "formal_safety",
        "silicon",
        None,
        ("committed-rtl", "report", "validator"),
        (),
        "",
        "bounded-safety",
    ),
    FacetSpec("ppa", "silicon", 5, _HARDWARE_REPORT, (), "silicon.ppa_report"),
    FacetSpec("physical", "silicon", None, _HARDWARE_REPORT, (), "", "measurement"),
)
FACET_BY_NAME: dict[str, FacetSpec] = {spec.name: spec for spec in FACETS}
INVALIDATION_MATRIX: dict[str, tuple[SubjectKind, ...]] = {
    spec.name: spec.subjects for spec in FACETS
}


def facets_invalidated_by(kind: SubjectKind) -> tuple[str, ...]:
    """Return the facets whose receipts a change of ``kind`` invalidates."""
    return tuple(name for name, kinds in INVALIDATION_MATRIX.items() if kind in kinds)


@dataclass(frozen=True, slots=True)
class Subject:
    """One content-addressed input of a facet receipt.

    Parameters
    ----------
    kind:
        Subject kind from :data:`SUBJECT_KINDS`.
    path:
        Repository-relative path (a file, or a directory for ``tree`` scope).
    sha256:
        Digest of the subject at recording time.
    scope:
        ``file`` (whole file), ``contract-sections`` (the descriptor sections
        that fix the model contract, so documentation edits do not invalidate
        evidence) or ``tree`` (every ``.py`` file under a directory).
    """

    kind: SubjectKind
    path: str
    sha256: str
    scope: SubjectScope = "file"

    def to_payload(self) -> dict[str, str]:
        """Return the JSON projection."""
        return {"kind": self.kind, "path": self.path, "sha256": self.sha256, "scope": self.scope}


@dataclass(frozen=True, slots=True)
class FacetReceipt:
    """One immutable record of an executed evidence command.

    Every field is a recorded fact of the run; none is derived by the verifier.
    """

    class_name: str
    facet: str
    profile: str
    claim_scope: str
    subjects: tuple[Subject, ...]
    evidence_refs: tuple[str, ...]
    command: tuple[str, ...]
    tool: Mapping[str, str]
    extra_tools: Mapping[str, str]
    runtime: Mapping[str, str]
    validator: Mapping[str, str]
    outcome: str
    exit_code: int
    counts: Mapping[str, int]
    recorded_at: str
    receipt_sha256: str = ""
    schema: str = FACET_RECEIPT_SCHEMA
    notes: str = ""
    artifacts: tuple[Subject, ...] = field(default_factory=tuple)

    def to_payload(self, *, sealed: bool = True) -> dict[str, Any]:
        """Return the JSON payload, with the seal digest when ``sealed``."""
        payload: dict[str, Any] = {
            "schema": self.schema,
            "class_name": self.class_name,
            "facet": self.facet,
            "profile": self.profile,
            "claim_scope": self.claim_scope,
            "subjects": [subject.to_payload() for subject in self.subjects],
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
            "evidence_refs": list(self.evidence_refs),
            "command": list(self.command),
            "tool": dict(self.tool),
            "extra_tools": dict(self.extra_tools),
            "runtime": dict(self.runtime),
            "validator": dict(self.validator),
            "outcome": self.outcome,
            "exit_code": self.exit_code,
            "counts": dict(self.counts),
            "recorded_at": self.recorded_at,
            "notes": self.notes,
        }
        if sealed:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload

    def sealed(self) -> FacetReceipt:
        """Return a copy whose ``receipt_sha256`` seals the current content."""
        digest = seal_digest(self.to_payload(sealed=False))
        return FacetReceipt(
            class_name=self.class_name,
            facet=self.facet,
            profile=self.profile,
            claim_scope=self.claim_scope,
            subjects=self.subjects,
            evidence_refs=self.evidence_refs,
            command=self.command,
            tool=self.tool,
            extra_tools=self.extra_tools,
            runtime=self.runtime,
            validator=self.validator,
            outcome=self.outcome,
            exit_code=self.exit_code,
            counts=self.counts,
            recorded_at=self.recorded_at,
            receipt_sha256=digest,
            schema=self.schema,
            notes=self.notes,
            artifacts=self.artifacts,
        )

    def subject_kinds(self) -> frozenset[str]:
        """Return the subject kinds the receipt carries."""
        return frozenset(subject.kind for subject in self.subjects)


def seal_digest(unsealed_payload: Mapping[str, Any]) -> str:
    """Return the seal digest of a receipt payload without ``receipt_sha256``."""
    if "receipt_sha256" in unsealed_payload:
        raise FacetReceiptError("seal input must not carry receipt_sha256")
    return sha256_canonical_json(unsealed_payload)


def _str_field(payload: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not value and not allow_empty):
        raise FacetReceiptError(f"receipt field {key!r} must be a non-empty string")
    return value


def _str_mapping(payload: Mapping[str, Any], key: str) -> dict[str, str]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise FacetReceiptError(f"receipt field {key!r} must map strings to strings")
    return dict(value)


def _str_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise FacetReceiptError(f"receipt field {key!r} must be a list of strings")
    return tuple(value)


def _int_field(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FacetReceiptError(f"receipt field {name!r} must be an integer")
    return value


def _parse_subject(raw: object, name: str) -> Subject:
    if not isinstance(raw, Mapping):
        raise FacetReceiptError(f"receipt {name} entries must be objects")
    kind = raw.get("kind")
    if kind not in SUBJECT_KINDS:
        raise FacetReceiptError(f"receipt {name} kind {kind!r} is not a subject kind")
    path = raw.get("path")
    if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
        raise FacetReceiptError(f"receipt {name} path must be repository-relative, got {path!r}")
    digest = raw.get("sha256")
    if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
        raise FacetReceiptError(f"receipt {name} {path!r} needs a SHA-256 digest")
    scope = raw.get("scope", "file")
    if scope not in {"file", "contract-sections", "tree"}:
        raise FacetReceiptError(f"receipt {name} scope {scope!r} is unknown")
    return Subject(
        kind=cast(SubjectKind, kind),
        path=path,
        sha256=digest,
        scope=cast(SubjectScope, scope),
    )


def parse_receipt(payload: Mapping[str, Any]) -> FacetReceipt:
    """Validate a receipt payload and return a :class:`FacetReceipt`.

    Structural validation only: the seal, the outcome and the subject
    freshness are judged by :func:`credit_problems` and the verifier.

    Raises
    ------
    FacetReceiptError
        If a required field is missing or malformed.
    """
    schema = _str_field(payload, "schema")
    if schema != FACET_RECEIPT_SCHEMA:
        raise FacetReceiptError(f"unsupported receipt schema {schema!r}")
    class_name = _str_field(payload, "class_name")
    if not _CLASS_NAME.fullmatch(class_name):
        raise FacetReceiptError(f"receipt class_name {class_name!r} is not an identifier")
    facet = _str_field(payload, "facet")
    if facet not in FACET_BY_NAME:
        raise FacetReceiptError(f"receipt facet {facet!r} is not a readiness facet")
    outcome = _str_field(payload, "outcome")
    if outcome not in OUTCOMES:
        raise FacetReceiptError(f"receipt outcome {outcome!r} is not one of {sorted(OUTCOMES)}")
    recorded_at = _str_field(payload, "recorded_at")
    if not _TIMESTAMP.fullmatch(recorded_at):
        raise FacetReceiptError("receipt recorded_at must be a UTC timestamp ending in Z")
    raw_subjects = payload.get("subjects")
    if not isinstance(raw_subjects, list):
        raise FacetReceiptError("receipt subjects must be a list")
    raw_artifacts = payload.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise FacetReceiptError("receipt artifacts must be a list")
    raw_counts = payload.get("counts")
    if not isinstance(raw_counts, Mapping):
        raise FacetReceiptError("receipt counts must be an object")
    counts = {str(key): _int_field(value, f"counts.{key}") for key, value in raw_counts.items()}
    for key in ("collected", "passed", "failed", "errors", "skipped"):
        counts.setdefault(key, 0)
    command = _str_tuple(payload, "command")
    if not command:
        raise FacetReceiptError("receipt command must not be empty")
    seal = payload.get("receipt_sha256", "")
    if not isinstance(seal, str) or (seal and not _SHA256_HEX.fullmatch(seal)):
        raise FacetReceiptError("receipt_sha256 must be a SHA-256 hex digest")
    return FacetReceipt(
        class_name=class_name,
        facet=facet,
        profile=_str_field(payload, "profile"),
        claim_scope=_str_field(payload, "claim_scope", allow_empty=True),
        subjects=tuple(_parse_subject(item, "subjects") for item in raw_subjects),
        evidence_refs=_str_tuple(payload, "evidence_refs"),
        command=command,
        tool=_str_mapping(payload, "tool"),
        extra_tools=_str_mapping(payload, "extra_tools"),
        runtime=_str_mapping(payload, "runtime"),
        validator=_str_mapping(payload, "validator"),
        outcome=outcome,
        exit_code=_int_field(payload.get("exit_code"), "exit_code"),
        counts=counts,
        recorded_at=recorded_at,
        receipt_sha256=seal,
        schema=schema,
        notes=_str_field(payload, "notes", allow_empty=True) if "notes" in payload else "",
        artifacts=tuple(_parse_subject(item, "artifacts") for item in raw_artifacts),
    )


def load_receipt(path: Path) -> FacetReceipt:
    """Load and structurally validate one receipt file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FacetReceiptError(f"cannot read receipt {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise FacetReceiptError(f"receipt {path} is not a JSON object")
    return parse_receipt(payload)


def credit_problems(receipt: FacetReceipt, *, class_name: str | None = None) -> tuple[str, ...]:
    """Return why a receipt cannot credit its facet; empty when it can.

    Freshness of the subjects is not judged here (it needs the repository);
    see :func:`sc_neurocore.neurons.readiness.verify_receipt`.

    Parameters
    ----------
    receipt:
        The receipt to judge.
    class_name:
        When given, the class the receipt is being read for; a receipt for a
        different class is a wrong-subject receipt.
    """
    problems: list[str] = []
    spec = FACET_BY_NAME.get(receipt.facet)
    if spec is None:
        return (f"unknown facet {receipt.facet!r}",)
    if class_name is not None and receipt.class_name != class_name:
        problems.append(f"receipt is for {receipt.class_name}, not {class_name}")
    if not receipt.receipt_sha256:
        problems.append("receipt is not sealed")
    elif receipt.receipt_sha256 != seal_digest(receipt.to_payload(sealed=False)):
        problems.append("receipt seal does not match its content")
    if spec.claim_scope and receipt.claim_scope != spec.claim_scope:
        problems.append(
            f"facet {spec.name} requires claim scope {spec.claim_scope!r}, "
            f"receipt states {receipt.claim_scope!r}"
        )
    if receipt.outcome != "passed":
        problems.append(f"outcome is {receipt.outcome!r}, not 'passed'")
    if receipt.exit_code != 0:
        problems.append(f"exit code {receipt.exit_code} is not 0")
    counts = receipt.counts
    if counts.get("passed", 0) < 1:
        problems.append("no passed check recorded")
    for key in ("failed", "errors", "skipped"):
        if counts.get(key, 0) > 0:
            problems.append(f"{counts[key]} {key} check(s) recorded")
    present = receipt.subject_kinds()
    for kind in spec.required_subjects:
        if kind not in present:
            problems.append(f"required subject kind {kind!r} is missing")
    for subject in receipt.subjects:
        if subject.kind not in spec.subjects:
            problems.append(f"subject kind {subject.kind!r} is not an input of {spec.name}")
    return tuple(problems)


def receipt_filename(class_name: str, facet: str, recorded_at: str) -> str:
    """Return the canonical receipt file name for one run."""
    stamp = recorded_at.replace("-", "").replace(":", "")
    return f"{class_name}__{facet.replace(':', '-')}__{stamp}.json"


def iter_receipts(directory: Path = RECEIPT_DIR) -> Iterator[tuple[Path, FacetReceipt]]:
    """Yield every receipt file in ``directory`` in file-name order.

    Raises
    ------
    FacetReceiptError
        If any receipt file is malformed; a broken receipt is an error, not a
        silently ignored file.
    """
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        yield path, load_receipt(path)


def latest_receipts(
    directory: Path = RECEIPT_DIR,
) -> dict[tuple[str, str], tuple[Path, FacetReceipt]]:
    """Return the newest receipt per ``(class_name, facet)``.

    Newest is decided by ``recorded_at`` and then by file name, so an
    append-only successor always supersedes its predecessor.
    """
    latest: dict[tuple[str, str], tuple[Path, FacetReceipt]] = {}
    for path, receipt in iter_receipts(directory):
        key = (receipt.class_name, receipt.facet)
        current = latest.get(key)
        if current is None or (receipt.recorded_at, path.name) > (
            current[1].recorded_at,
            current[0].name,
        ):
            latest[key] = (path, receipt)
    return latest


_CONTRACT_SECTIONS = ("state", "parameters", "integration", "dynamics")
_CONTRACT_METADATA = ("name", "class_name", "module")


def descriptor_contract_digest(payload: Mapping[str, Any]) -> str:
    """Return the digest of the descriptor sections that fix the model contract.

    Only identity (``metadata.name``/``class_name``/``module``), state,
    parameters, integration and dynamics take part, so a documentation,
    provenance or evidence edit never invalidates a receipt while a changed
    equation, parameter default, dt or method always does.
    """
    metadata = payload.get("metadata", {})
    contract: dict[str, Any] = {
        "metadata": {
            key: metadata.get(key) for key in _CONTRACT_METADATA if isinstance(metadata, Mapping)
        }
    }
    for section in _CONTRACT_SECTIONS:
        contract[section] = payload.get(section, {})
    return sha256_canonical_json(contract)


def descriptor_contract_digest_of(path: Path) -> str:
    """Return :func:`descriptor_contract_digest` of a descriptor TOML file."""
    with path.open("rb") as handle:
        return descriptor_contract_digest(tomllib.load(handle))


__all__ = [
    "FACETS",
    "FACET_BY_NAME",
    "FACET_RECEIPT_SCHEMA",
    "INVALIDATION_MATRIX",
    "OUTCOMES",
    "RECEIPT_DIR",
    "SUBJECT_KINDS",
    "FacetReceipt",
    "FacetReceiptError",
    "FacetSpec",
    "Subject",
    "credit_problems",
    "descriptor_contract_digest",
    "descriptor_contract_digest_of",
    "facets_invalidated_by",
    "iter_receipts",
    "latest_receipts",
    "load_receipt",
    "parse_receipt",
    "receipt_filename",
    "seal_digest",
]
