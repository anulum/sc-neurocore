# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence-bound readiness: declared versus verified

"""Per-model readiness verification: declared metadata versus bound evidence.

:mod:`sc_neurocore.neurons.descriptor_tiers` scores the readiness a descriptor
*declares* (tier semantics v1: a boolean flag plus a non-empty evidence string
credits a rung). This module keeps that declared score unchanged and adds the
*verified* score next to it. A facet is verified only through a facet receipt
(:mod:`sc_neurocore.neurons.facet_receipts`) whose subjects still match the
repository; every other declared facet is reported by how far its evidence can
be followed:

``not-declared``
    the descriptor does not claim the facet;
``declared``
    claimed, but the evidence field names nothing that can be located (prose,
    an inline configuration, or no field for the facet);
``unavailable``
    claimed, and at least one named file or test node does not exist;
``located``
    every named file and test node exists, but no receipt records a run;
``bound``
    the newest receipt is creditable and every subject digest still matches;
``stale``
    the newest receipt was creditable when recorded, but a subject changed;
``invalid``
    the newest receipt cannot credit the facet (failed, skipped, unsealed,
    wrong class, wrong claim scope or missing required subjects).

Verified tiers climb only over ``bound`` facets that the descriptor also
declares, one rung at a time, so no receipt can credit a tier the descriptor
does not claim and no declaration can credit a tier without a receipt.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from sc_neurocore.neurons.descriptor_tiers import science_tier, silicon_tier
from sc_neurocore.neurons.evidence_references import (
    EvidenceReference,
    parse_evidence_field,
    sha256_file,
    sha256_tree,
)
from sc_neurocore.neurons.facet_receipts import (
    FACET_BY_NAME,
    FACETS,
    RECEIPT_DIR,
    FacetReceipt,
    FacetSpec,
    Subject,
    SubjectKind,
    credit_problems,
    descriptor_contract_digest,
    descriptor_contract_digest_of,
    latest_receipts,
)
from sc_neurocore.neurons.model_catalogue import descriptor_path, load_descriptor_payload
from sc_neurocore.neurons.model_descriptor import (
    ModelDescriptor,
    descriptor_completeness_tier,
    parse_model_descriptor,
)
from sc_neurocore.neurons.model_identity import identity_registry

REPO_ROOT = Path(__file__).resolve().parents[3]
FacetStatus = Literal[
    "not-declared", "declared", "unavailable", "located", "bound", "stale", "invalid"
]
FACET_STATUSES: tuple[FacetStatus, ...] = (
    "not-declared",
    "declared",
    "unavailable",
    "located",
    "bound",
    "stale",
    "invalid",
)

_COMPILER_TREE = "src/sc_neurocore/compiler"
_COMPILER_FILES = (
    "src/sc_neurocore/neurons/universal_dsl.py",
    "src/sc_neurocore/neurons/equation_builder.py",
    "src/sc_neurocore/neurons/equation_namespace.py",
    "src/sc_neurocore/neurons/equation_safety.py",
    "src/sc_neurocore/neurons/equation_units_runtime.py",
    "src/sc_neurocore/neurons/expression_derivative.py",
    "src/sc_neurocore/neurons/schema_validator.py",
    "src/sc_neurocore/neurons/schema_contracts.py",
)


@dataclass(frozen=True, slots=True)
class FacetVerification:
    """Verification outcome of one facet for one model."""

    facet: str
    declared: bool
    status: FacetStatus
    evidence: tuple[EvidenceReference, ...]
    receipt: str
    changed_subjects: tuple[str, ...]
    problems: tuple[str, ...]

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-compatible projection.

        The parsed evidence references are listed once per descriptor field on
        the owning :class:`ReadinessRecord`; here the facet names its field.
        """
        return {
            "facet": self.facet,
            "declared": self.declared,
            "status": self.status,
            "evidence_field": FACET_BY_NAME[self.facet].evidence_field,
            "receipt": self.receipt,
            "changed_subjects": list(self.changed_subjects),
            "problems": list(self.problems),
        }


@dataclass(frozen=True, slots=True)
class ReadinessRecord:
    """Declared and verified readiness of one registered class."""

    class_name: str
    kind: str
    has_descriptor: bool
    declared_science: int
    declared_silicon: int | None
    verified_science: int
    verified_silicon: int | None
    facets: tuple[FacetVerification, ...]

    @property
    def declared_science_label(self) -> str:
        """Declared science tier as ``S<n>``."""
        return f"S{self.declared_science}"

    @property
    def declared_silicon_label(self) -> str:
        """Declared silicon tier as ``H<n>`` or ``none``."""
        return "none" if self.declared_silicon is None else f"H{self.declared_silicon}"

    @property
    def verified_science_label(self) -> str:
        """Verified science tier as ``S<n>``."""
        return f"S{self.verified_science}"

    @property
    def verified_silicon_label(self) -> str:
        """Verified silicon tier as ``H<n>`` or ``none``."""
        return "none" if self.verified_silicon is None else f"H{self.verified_silicon}"

    def facet(self, name: str) -> FacetVerification:
        """Return the verification of one facet by name."""
        for verification in self.facets:
            if verification.facet == name:
                return verification
        raise KeyError(name)

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-compatible projection with a stable field order."""
        return {
            "class_name": self.class_name,
            "kind": self.kind,
            "has_descriptor": self.has_descriptor,
            "declared": {
                "science_tier": self.declared_science,
                "science_label": self.declared_science_label,
                "silicon_tier": self.declared_silicon,
                "silicon_label": self.declared_silicon_label,
            },
            "verified": {
                "science_tier": self.verified_science,
                "science_label": self.verified_science_label,
                "silicon_tier": self.verified_silicon,
                "silicon_label": self.verified_silicon_label,
            },
            "evidence": self.evidence_by_field(),
            "facets": [verification.to_public_dict() for verification in self.facets],
        }

    def evidence_by_field(self) -> dict[str, list[dict[str, str]]]:
        """Return the parsed references of every non-empty descriptor evidence field."""
        fields: dict[str, list[dict[str, str]]] = {}
        for verification in self.facets:
            field = FACET_BY_NAME[verification.facet].evidence_field
            if field and field not in fields and verification.evidence:
                fields[field] = [reference.to_public_dict() for reference in verification.evidence]
        return fields


def declared_facets(descriptor: ModelDescriptor) -> dict[str, bool]:
    """Return which facets a descriptor declares, under tier semantics v1.

    The predicates are exactly the anchors :mod:`descriptor_tiers` credits, so
    a declared facet here is a declared rung there.
    """
    validation = descriptor.validation
    silicon = descriptor.silicon
    declared: dict[str, bool] = {
        "dynamics_faithful": bool(descriptor.dynamics) and validation.dynamics_faithful,
        "class_validated": validation.is_class_validated,
        "rtl_compile": silicon.compiles,
        "cosim": silicon.cosim_validated and bool(silicon.cosim_evidence),
        "synthesis": silicon.synthesised and bool(silicon.synth_report),
        "timing": (
            silicon.timing_closed and bool(silicon.timing_report) and silicon.clock_mhz is not None
        ),
        "formal_equivalence": silicon.formally_equivalent and bool(silicon.equivalence_proof),
        "formal_safety": False,
        "ppa": silicon.ppa_signed and bool(silicon.ppa_report),
        "physical": False,
    }
    implemented = {
        backend.name for backend in descriptor.backends if backend.status == "implemented"
    }
    for backend in ("python", "rust", "julia", "go", "mojo"):
        declared[f"backend:{backend}"] = backend in implemented
    return declared


def facet_evidence_field(descriptor: ModelDescriptor, spec: FacetSpec) -> str:
    """Return the raw evidence text the descriptor holds for ``spec``."""
    if spec.evidence_field == "validation.evidence":
        return descriptor.validation.evidence
    if spec.evidence_field.startswith("silicon."):
        return str(getattr(descriptor.silicon, spec.evidence_field.split(".", 1)[1]))
    return ""


def _tree_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))


@lru_cache(maxsize=8)
def compiler_subjects(repo_root: Path = REPO_ROOT) -> tuple[Subject, ...]:
    """Return the compiler subjects shared by every generated-RTL receipt."""
    subjects: list[Subject] = []
    tree = repo_root / _COMPILER_TREE
    if tree.is_dir():
        subjects.append(
            Subject("compiler", _COMPILER_TREE, sha256_tree(_tree_files(tree), repo_root), "tree")
        )
    for relative in _COMPILER_FILES:
        path = repo_root / relative
        if path.is_file():
            subjects.append(Subject("compiler", relative, sha256_file(path)))
    return tuple(subjects)


def current_digest(subject: Subject, repo_root: Path = REPO_ROOT) -> str | None:
    """Recompute a subject's digest from the current repository, ``None`` if absent."""
    path = repo_root / subject.path
    if subject.scope == "tree":
        if not path.is_dir():
            return None
        return sha256_tree(_tree_files(path), repo_root)
    if not path.is_file():
        return None
    if subject.scope == "contract-sections":
        return descriptor_contract_digest_of(path)
    return sha256_file(path)


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def derive_subjects(
    class_name: str,
    facet: str,
    *,
    repo_root: Path = REPO_ROOT,
    evidence_refs: Iterable[str] = (),
    extra_subjects: Iterable[Subject] = (),
) -> tuple[Subject, ...]:
    """Derive the current subjects a receipt for ``facet`` must cover.

    Parameters
    ----------
    class_name:
        Registered class the receipt is for.
    facet:
        Facet name.
    repo_root:
        Repository root.
    evidence_refs:
        Evidence references to take validator and report subjects from; the
        descriptor's own field for the facet is always included.
    extra_subjects:
        Subjects the caller adds: the committed RTL a formal or synthesis lane
        covers (the recorder resolves it through the formal inventory tool)
        or a native backend source the registry cannot derive.

    Returns
    -------
    tuple[Subject, ...]
        Subjects with digests of the current content, de-duplicated by
        ``(kind, path)`` and sorted.
    """
    spec = next(item for item in FACETS if item.name == facet)
    identity = identity_registry()[class_name]
    payload = load_descriptor_payload(class_name)
    subjects: dict[tuple[str, str], Subject] = {}

    def add(subject: Subject) -> None:
        subjects.setdefault((subject.kind, subject.path), subject)

    def add_file(kind: SubjectKind, relative: str) -> None:
        path = repo_root / relative
        if path.is_file():
            add(Subject(kind, relative, sha256_file(path)))

    descriptor_file = _relative(descriptor_path(class_name), repo_root)
    if payload is not None:
        add(
            Subject(
                "descriptor-contract",
                descriptor_file,
                descriptor_contract_digest(payload),
                "contract-sections",
            )
        )
    add_file("model-module", f"src/sc_neurocore/neurons/models/{identity.module}.py")
    for profile in identity.schema_profiles:
        for suffix in (".toml", ".json"):
            add_file(
                "schema-profile", f"src/sc_neurocore/neurons/model_schemas/{profile.stem}{suffix}"
            )
    refs = list(evidence_refs)
    if payload is not None:
        descriptor = parse_model_descriptor(payload)
        refs.append(facet_evidence_field(descriptor, spec))
        reproducibility = descriptor.reproducibility
        for candidate in (reproducibility.reference_config, reproducibility.golden_citation):
            for reference in parse_evidence_field(candidate, repo_root):
                if reference.kind == "artifact-file" and reference.is_resolved:
                    add_file("source-reference", reference.path)
        for reference in parse_evidence_field(descriptor.validation.evidence, repo_root):
            if reference.kind == "artifact-file" and reference.is_resolved:
                add_file("source-reference", reference.path)
    for raw in refs:
        for reference in parse_evidence_field(raw, repo_root):
            if not reference.is_resolved:
                continue
            if reference.kind in {"test-node", "test-file"}:
                add_file("validator", reference.path)
            elif reference.kind == "artifact-file":
                kind: SubjectKind = (
                    "report" if "report" in spec.required_subjects else "source-reference"
                )
                add_file(kind, reference.path)
    if "compiler" in spec.subjects:
        for subject in compiler_subjects(repo_root):
            add(subject)
    for subject in extra_subjects:
        add(subject)
    return tuple(
        subject for _key, subject in sorted(subjects.items()) if subject.kind in spec.subjects
    )


def verify_receipt(
    receipt: FacetReceipt,
    *,
    class_name: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[FacetStatus, tuple[str, ...], tuple[str, ...]]:
    """Judge one receipt against the current repository.

    Returns
    -------
    tuple[FacetStatus, tuple[str, ...], tuple[str, ...]]
        ``(status, changed_subject_paths, problems)`` where status is
        ``invalid`` (cannot credit), ``stale`` (creditable but a subject
        changed or vanished) or ``bound``.
    """
    problems = credit_problems(receipt, class_name=class_name)
    if problems:
        return "invalid", (), problems
    changed: list[str] = []
    for subject in receipt.subjects:
        digest = current_digest(subject, repo_root)
        if digest is None:
            changed.append(f"{subject.kind}:{subject.path} (missing)")
        elif digest != subject.sha256:
            changed.append(f"{subject.kind}:{subject.path}")
    if changed:
        return "stale", tuple(changed), ()
    return "bound", (), ()


def _facet_verification(
    spec: FacetSpec,
    *,
    declared: bool,
    evidence: tuple[EvidenceReference, ...],
    receipt_entry: tuple[Path, FacetReceipt] | None,
    class_name: str,
    repo_root: Path,
) -> FacetVerification:
    receipt_name = receipt_entry[0].name if receipt_entry is not None else ""
    if not declared:
        return FacetVerification(spec.name, False, "not-declared", evidence, receipt_name, (), ())
    if receipt_entry is not None:
        status, changed, problems = verify_receipt(
            receipt_entry[1], class_name=class_name, repo_root=repo_root
        )
        return FacetVerification(spec.name, True, status, evidence, receipt_name, changed, problems)
    locatable = [reference for reference in evidence if reference.is_locatable]
    if not locatable:
        return FacetVerification(spec.name, True, "declared", evidence, "", (), ())
    unresolved = tuple(
        f"{reference.resolution}: {reference.raw}"
        for reference in locatable
        if not reference.is_resolved
    )
    if unresolved:
        return FacetVerification(spec.name, True, "unavailable", evidence, "", (), unresolved)
    return FacetVerification(spec.name, True, "located", evidence, "", (), ())


def _verified_science(kernel: int, bound: Mapping[str, bool]) -> int:
    if kernel < 3:
        return kernel
    if not bound["dynamics_faithful"]:
        return 3
    if not bound["class_validated"]:
        return 4
    return 5


def _verified_silicon(bound: Mapping[str, bool]) -> int | None:
    if not (bound["rtl_compile"] or bound["cosim"]):
        return None
    tier = 0
    for facet in ("cosim", "synthesis", "timing", "formal_equivalence", "ppa"):
        if not bound[facet]:
            break
        tier += 1
    return tier


def verify_model(
    class_name: str,
    *,
    repo_root: Path = REPO_ROOT,
    receipts: Mapping[tuple[str, str], tuple[Path, FacetReceipt]] | None = None,
) -> ReadinessRecord:
    """Verify every facet of one registered class.

    Parameters
    ----------
    class_name:
        Registered class name (aliases are resolved by the caller).
    repo_root:
        Repository root the evidence paths are relative to.
    receipts:
        Newest receipts per ``(class_name, facet)``; read from
        :data:`~sc_neurocore.neurons.facet_receipts.RECEIPT_DIR` when omitted.
    """
    if receipts is None:
        receipts = latest_receipts()
    identity = identity_registry()[class_name]
    payload = load_descriptor_payload(class_name)
    if payload is None:
        facets = tuple(
            FacetVerification(spec.name, False, "not-declared", (), "", (), ()) for spec in FACETS
        )
        return ReadinessRecord(class_name, identity.kind, False, 0, None, 0, None, facets)
    descriptor = parse_model_descriptor(payload)
    declared = declared_facets(descriptor)
    verifications: list[FacetVerification] = []
    for spec in FACETS:
        evidence = parse_evidence_field(facet_evidence_field(descriptor, spec), repo_root)
        verifications.append(
            _facet_verification(
                spec,
                declared=declared[spec.name],
                evidence=evidence,
                receipt_entry=receipts.get((class_name, spec.name)),
                class_name=class_name,
                repo_root=repo_root,
            )
        )
    bound = {item.facet: item.status == "bound" for item in verifications}
    return ReadinessRecord(
        class_name=class_name,
        kind=identity.kind,
        has_descriptor=True,
        declared_science=science_tier(descriptor),
        declared_silicon=silicon_tier(descriptor),
        verified_science=_verified_science(descriptor_completeness_tier(descriptor), bound),
        verified_silicon=_verified_silicon(bound),
        facets=tuple(verifications),
    )


def readiness_report(
    *,
    repo_root: Path = REPO_ROOT,
    receipt_dir: Path = RECEIPT_DIR,
) -> dict[str, ReadinessRecord]:
    """Verify every registered class (aliases excluded), keyed by class name."""
    receipts = latest_receipts(receipt_dir)
    return {
        name: verify_model(name, repo_root=repo_root, receipts=receipts)
        for name, identity in identity_registry().items()
        if identity.kind != "api-alias"
    }


def summarise(records: Iterable[ReadinessRecord]) -> dict[str, Any]:
    """Return count summaries of declared versus verified readiness."""
    records = tuple(records)
    facet_status: dict[str, dict[str, int]] = {spec.name: {} for spec in FACETS}
    declared_science: dict[str, int] = {}
    verified_science: dict[str, int] = {}
    declared_silicon: dict[str, int] = {}
    verified_silicon: dict[str, int] = {}
    for record in records:
        declared_science[record.declared_science_label] = (
            declared_science.get(record.declared_science_label, 0) + 1
        )
        verified_science[record.verified_science_label] = (
            verified_science.get(record.verified_science_label, 0) + 1
        )
        declared_silicon[record.declared_silicon_label] = (
            declared_silicon.get(record.declared_silicon_label, 0) + 1
        )
        verified_silicon[record.verified_silicon_label] = (
            verified_silicon.get(record.verified_silicon_label, 0) + 1
        )
        for verification in record.facets:
            bucket = facet_status[verification.facet]
            bucket[verification.status] = bucket.get(verification.status, 0) + 1
    return {
        "models": len(records),
        "declared_science_tiers": dict(sorted(declared_science.items())),
        "verified_science_tiers": dict(sorted(verified_science.items())),
        "declared_silicon_tiers": dict(sorted(declared_silicon.items())),
        "verified_silicon_tiers": dict(sorted(verified_silicon.items())),
        "facet_statuses": {
            name: dict(sorted(statuses.items())) for name, statuses in facet_status.items()
        },
    }


__all__ = [
    "FACET_STATUSES",
    "REPO_ROOT",
    "FacetStatus",
    "FacetVerification",
    "ReadinessRecord",
    "compiler_subjects",
    "current_digest",
    "declared_facets",
    "derive_subjects",
    "facet_evidence_field",
    "readiness_report",
    "summarise",
    "verify_model",
    "verify_receipt",
]
