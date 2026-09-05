# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-profile inventory and validator registry

"""Per-(class, profile) inventory and the validator registry it carries.

The identity registry (:mod:`sc_neurocore.neurons.model_identity`) binds every
schema stem to exactly one class. The model profile
(:mod:`sc_neurocore.neurons.model_profile`) separates a schema into scientific
model, numerical realisation and lowering profile. Readiness
(:mod:`sc_neurocore.neurons.readiness`) verifies a descriptor's declared
facets per profile. This module joins the three into one row per bound
profile so a reader can see, for each identity and each of its numerical
profiles, what the science is, how it is advanced, what can be lowered, which
validators the descriptor and the schema declare for each facet, which of them
resolve and are executed into a bound receipt, and whether the profile is
admitted for evidence at all.

Admission is the gate FF-04, FF-05 and FF-14 consume: a profile is
``admitted`` only when it is executable, contradiction-free and every facet the
descriptor declares through an evidence field has at least one executable
validator (a test file or exact test node that exists) scoped to that profile.
Descriptor evidence fields are class-scoped, so they validate the class's
canonical profile; a non-canonical profile is validated only by its own schema
``[validation].evidence`` or by a receipt recorded for it, never by the
canonical profile's tests. A declared facet backed by prose alone blocks
admission; a descriptor flag never admits a scientific row on its own. Backend
facets have no descriptor evidence field yet (owned by the native admission
work), so declared-but-unvalidated backends are reported on the row instead of
deciding admission.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sc_neurocore.neurons.evidence_references import EvidenceReference, parse_evidence_field
from sc_neurocore.neurons.facet_receipts import FACET_BY_NAME, FACETS, FacetSpec
from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.model_descriptor import ModelDescriptor, parse_model_descriptor
from sc_neurocore.neurons.model_identity import (
    ModelIdentity,
    identity_registry,
    schema_for_class,
)
from sc_neurocore.neurons.model_profile import METHOD_TABLE, ModelProfile, resolve_profile
from sc_neurocore.neurons.readiness import (
    REPO_ROOT,
    FacetStatus,
    ReadinessRecord,
    declared_facets,
    facet_evidence_field,
    verify_model,
)
from sc_neurocore.neurons.universal_dsl import load_schema

AdmissionStatus = Literal["admitted", "blocked", "not-executable"]
ValidatorOrigin = Literal["descriptor", "schema"]
ValidatorScope = Literal["profile", "class"]


@dataclass(frozen=True, slots=True)
class ValidatorBinding:
    """One declared validator bound to one facet of one profile.

    Parameters
    ----------
    facet:
        Facet name the validator is declared for.
    origin:
        ``descriptor`` (the descriptor's evidence field for the facet) or
        ``schema`` (the schema's ``[validation].evidence``).
    scope:
        ``profile`` when the validator belongs to this profile (a schema
        validator, or a descriptor validator on the class's canonical profile);
        ``class`` when it is a descriptor validator listed on a non-canonical
        profile, where it documents the class but cannot admit the profile.
    reference:
        Parsed and resolved evidence reference.
    """

    facet: str
    origin: ValidatorOrigin
    scope: ValidatorScope
    reference: EvidenceReference

    @property
    def executable(self) -> bool:
        """Whether the reference names an existing test file or test node."""
        return self.reference.kind in {"test-file", "test-node"} and self.reference.is_resolved

    @property
    def admits(self) -> bool:
        """Whether the validator can admit this profile (executable and profile-scoped)."""
        return self.executable and self.scope == "profile"

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "facet": self.facet,
            "origin": self.origin,
            "scope": self.scope,
            "executable": self.executable,
            **self.reference.to_public_dict(),
        }


@dataclass(frozen=True, slots=True)
class FacetRegistryEntry:
    """Validator registry of one facet for one profile.

    Parameters
    ----------
    facet:
        Facet name.
    declared:
        Whether the descriptor declares the facet.
    status:
        Readiness status of the facet for this profile.
    receipt:
        Newest receipt file name for this profile, empty when none.
    validators:
        Every declared validator for the facet, executable or not.
    """

    facet: str
    declared: bool
    status: FacetStatus
    receipt: str
    validators: tuple[ValidatorBinding, ...]

    @property
    def executable_validators(self) -> tuple[ValidatorBinding, ...]:
        """Validators that name an existing test file or node."""
        return tuple(binding for binding in self.validators if binding.executable)

    @property
    def admitting_validators(self) -> tuple[ValidatorBinding, ...]:
        """Executable validators scoped to this profile."""
        return tuple(binding for binding in self.validators if binding.admits)

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "facet": self.facet,
            "declared": self.declared,
            "status": self.status,
            "receipt": self.receipt,
            "validators": [binding.to_public_dict() for binding in self.validators],
        }


@dataclass(frozen=True, slots=True)
class ProfileRow:
    """One (class, profile) row of the inventory.

    Parameters
    ----------
    class_name:
        Registered class the profile is bound to.
    stem:
        Schema stem of the profile.
    identity_kind:
        Identity kind of the class.
    counts_in_source_catalogue:
        Whether the class counts in the public source catalogue.
    canonical:
        Whether this stem is the class's canonical profile (the one Studio and
        the generators use).
    profile:
        Resolved model profile.
    descriptor_method, descriptor_dt:
        The descriptor's own integration label and step (the hand class's
        numerics), empty and ``None`` without a descriptor.
    method_agreement:
        ``same`` when the descriptor label equals the profile method,
        ``differs`` when it names another realisation (a hand class may keep a
        different default than its schema profile), ``no-descriptor`` otherwise.
    readiness:
        Readiness record verified for exactly this profile.
    registry:
        Validator registry per facet.
    admission:
        ``admitted``, ``blocked`` or ``not-executable``.
    admission_reasons:
        Why the profile is not admitted, empty when admitted.
    unvalidated_backends:
        Backends the descriptor declares implemented without any validator
        field to bind them; reported, not decided, until the native admission
        contract defines the field.
    """

    class_name: str
    stem: str
    identity_kind: str
    counts_in_source_catalogue: bool
    canonical: bool
    profile: ModelProfile
    descriptor_method: str
    descriptor_dt: float | None
    method_agreement: str
    readiness: ReadinessRecord
    registry: tuple[FacetRegistryEntry, ...]
    admission: AdmissionStatus
    admission_reasons: tuple[str, ...]
    unvalidated_backends: tuple[str, ...] = ()

    def entry(self, facet: str) -> FacetRegistryEntry:
        """Return the registry entry of one facet by name."""
        for item in self.registry:
            if item.facet == facet:
                return item
        raise KeyError(facet)

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection with a stable field order."""
        numerical = self.profile.numerical
        return {
            "class_name": self.class_name,
            "stem": self.stem,
            "identity_kind": self.identity_kind,
            "counts_in_source_catalogue": self.counts_in_source_catalogue,
            "canonical": self.canonical,
            "realisation_kind": self.profile.realisation_kind,
            "authored_profile": self.profile.authored,
            "scientific": {
                "name": self.profile.scientific.name,
                "doi": self.profile.scientific.doi,
                "biological_state": [v.name for v in self.profile.scientific.biological_state],
                "source_parameters": [p.name for p in self.profile.scientific.source_parameters],
                "equations": dict(self.profile.scientific.equations),
            },
            "numerical": {
                "method": numerical.method,
                "family": numerical.family,
                "exactness": numerical.exactness,
                "exactness_claimed": numerical.exactness_claimed,
                "dt": numerical.dt,
                "time_unit": numerical.time_unit,
                "substeps": numerical.substeps,
                "substep_kind": numerical.substep_kind,
                "macro_step": numerical.macro_step,
                "evaluation_order": list(numerical.evaluation_order),
                "auxiliary_registers": [r.name for r in numerical.auxiliary_registers],
                "implementation_parameters": [p.name for p in numerical.implementation_parameters],
                "timebase_parameters": [p.name for p in numerical.timebase_parameters],
                "admissible_methods": list(numerical.admissible_methods),
                "randomness": numerical.randomness.to_public_dict(),
            },
            "lowering": self.profile.lowering.to_public_dict(),
            "descriptor": {
                "method": self.descriptor_method,
                "dt": self.descriptor_dt,
                "method_agreement": self.method_agreement,
            },
            "profile_problems": list(self.profile.problems),
            "verified": {
                "science_label": self.readiness.verified_science_label,
                "silicon_label": self.readiness.verified_silicon_label,
            },
            "declared": {
                "science_label": self.readiness.declared_science_label,
                "silicon_label": self.readiness.declared_silicon_label,
            },
            "admission": self.admission,
            "admission_reasons": list(self.admission_reasons),
            "unvalidated_backends": list(self.unvalidated_backends),
            "registry": [entry.to_public_dict() for entry in self.registry],
        }


def _schema_validators(schema: dict[str, Any], repo_root: Path) -> tuple[EvidenceReference, ...]:
    validation = schema.get("validation")
    if not isinstance(validation, dict):
        return ()
    evidence = validation.get("evidence")
    if not isinstance(evidence, str) or not evidence:
        return ()
    return parse_evidence_field(evidence, repo_root)


def _facet_registry(
    spec: FacetSpec,
    *,
    descriptor: ModelDescriptor | None,
    declared: bool,
    canonical: bool,
    readiness: ReadinessRecord,
    schema_refs: tuple[EvidenceReference, ...],
    repo_root: Path,
) -> FacetRegistryEntry:
    verification = readiness.facet(spec.name)
    validators: list[ValidatorBinding] = []
    descriptor_scope: ValidatorScope = "profile" if canonical else "class"
    if descriptor is not None:
        for reference in parse_evidence_field(facet_evidence_field(descriptor, spec), repo_root):
            validators.append(
                ValidatorBinding(spec.name, "descriptor", descriptor_scope, reference)
            )
    if spec.axis == "science":
        for reference in schema_refs:
            validators.append(ValidatorBinding(spec.name, "schema", "profile", reference))
    return FacetRegistryEntry(
        facet=spec.name,
        declared=declared,
        status=verification.status,
        receipt=verification.receipt,
        validators=tuple(validators),
    )


def _admission(
    profile: ModelProfile, registry: Iterable[FacetRegistryEntry], *, canonical: bool
) -> tuple[AdmissionStatus, tuple[str, ...]]:
    if not profile.is_executable:
        return "not-executable", tuple(profile.lowering.limits)
    reasons: list[str] = list(profile.problems)
    for entry in registry:
        spec = FACET_BY_NAME[entry.facet]
        if not entry.declared or not spec.evidence_field:
            continue
        if entry.status == "bound":
            continue
        if entry.admitting_validators:
            continue
        if entry.executable_validators and not canonical:
            reasons.append(
                f"declared facet {entry.facet} is validated only by class-scoped tests of the "
                f"canonical profile; profile {profile.stem!r} has no validator or receipt of its own"
            )
        else:
            reasons.append(
                f"declared facet {entry.facet} has no executable validator "
                "(a descriptor flag or prose cannot admit it)"
            )
    return ("blocked" if reasons else "admitted"), tuple(reasons)


def profile_row(
    identity: ModelIdentity,
    stem: str,
    *,
    repo_root: Path = REPO_ROOT,
    receipts: Any = None,
) -> ProfileRow:
    """Build the inventory row of one bound profile.

    Parameters
    ----------
    identity:
        Identity record of the class.
    stem:
        One of the identity's bound schema stems.
    repo_root:
        Repository root the evidence paths are relative to.
    receipts:
        Newest receipts per ``(class, facet, profile)`` as returned by
        :func:`~sc_neurocore.neurons.facet_receipts.latest_receipts`; read
        from the receipt store when omitted.
    """
    schema = load_schema(stem)
    profile = resolve_profile(schema, stem=stem)
    payload = load_descriptor_payload(identity.class_name)
    descriptor = parse_model_descriptor(payload) if payload is not None else None
    readiness = verify_model(
        identity.class_name, repo_root=repo_root, receipts=receipts, profile=stem
    )
    declared = declared_facets(descriptor) if descriptor is not None else {}
    schema_refs = _schema_validators(schema, repo_root)
    canonical = schema_for_class(identity.class_name) == stem
    registry = tuple(
        _facet_registry(
            spec,
            descriptor=descriptor,
            declared=bool(declared.get(spec.name, False)),
            canonical=canonical,
            readiness=readiness,
            schema_refs=schema_refs,
            repo_root=repo_root,
        )
        for spec in FACETS
    )
    admission, reasons = _admission(profile, registry, canonical=canonical)
    unvalidated_backends = tuple(
        entry.facet.split(":", 1)[1]
        for entry in registry
        if entry.facet.startswith("backend:") and entry.declared and entry.status != "bound"
    )
    descriptor_method = descriptor.integration_method if descriptor is not None else ""
    descriptor_dt = descriptor.dt if descriptor is not None else None
    if descriptor is None:
        agreement = "no-descriptor"
    elif descriptor_method == profile.numerical.method:
        agreement = "same"
    else:
        agreement = "differs"
    return ProfileRow(
        class_name=identity.class_name,
        stem=stem,
        identity_kind=identity.kind,
        counts_in_source_catalogue=identity.counts_in_source_catalogue,
        canonical=canonical,
        profile=profile,
        descriptor_method=descriptor_method,
        descriptor_dt=descriptor_dt,
        method_agreement=agreement,
        readiness=readiness,
        registry=registry,
        admission=admission,
        admission_reasons=reasons,
        unvalidated_backends=unvalidated_backends,
    )


def profile_inventory(*, repo_root: Path = REPO_ROOT) -> tuple[ProfileRow, ...]:
    """Return one row per bound (class, profile), sorted by class and stem.

    Parameters
    ----------
    repo_root:
        Repository root the evidence paths are relative to.
    """
    from sc_neurocore.neurons.facet_receipts import latest_receipts

    receipts = latest_receipts()
    rows: list[ProfileRow] = []
    for identity in identity_registry().values():
        if identity.kind == "api-alias":
            continue
        for bound in identity.schema_profiles:
            rows.append(profile_row(identity, bound.stem, repo_root=repo_root, receipts=receipts))
    return tuple(sorted(rows, key=lambda row: (row.class_name, row.stem)))


def validator_registry(
    rows: Iterable[ProfileRow],
) -> dict[tuple[str, str], tuple[FacetRegistryEntry, ...]]:
    """Return the validator registry keyed by ``(class_name, stem)``."""
    return {(row.class_name, row.stem): row.registry for row in rows}


def summarise_inventory(rows: Iterable[ProfileRow]) -> dict[str, Any]:
    """Return count summaries over the inventory."""
    rows = tuple(rows)
    by_kind: dict[str, int] = {}
    by_method: dict[str, int] = {}
    by_exactness: dict[str, int] = {}
    by_admission: dict[str, int] = {}
    by_agreement: dict[str, int] = {}
    authored = 0
    multi_profile_classes = {
        name
        for name in {row.class_name for row in rows}
        if sum(1 for row in rows if row.class_name == name) > 1
    }
    for row in rows:
        by_kind[row.profile.realisation_kind] = by_kind.get(row.profile.realisation_kind, 0) + 1
        method = row.profile.numerical.method
        by_method[method] = by_method.get(method, 0) + 1
        exactness = row.profile.numerical.exactness
        by_exactness[exactness] = by_exactness.get(exactness, 0) + 1
        by_admission[row.admission] = by_admission.get(row.admission, 0) + 1
        by_agreement[row.method_agreement] = by_agreement.get(row.method_agreement, 0) + 1
        authored += int(row.profile.authored)
    return {
        "profiles": len(rows),
        "classes": len({row.class_name for row in rows}),
        "multi_profile_classes": sorted(multi_profile_classes),
        "authored_profiles": authored,
        "realisation_kinds": dict(sorted(by_kind.items())),
        "methods": dict(sorted(by_method.items())),
        "exactness": dict(sorted(by_exactness.items())),
        "admission": dict(sorted(by_admission.items())),
        "descriptor_method_agreement": dict(sorted(by_agreement.items())),
    }


def method_table() -> tuple[dict[str, str], ...]:
    """Return the method → exactness → family → lowering mapping table."""
    return tuple(dict(row) for row in METHOD_TABLE)


__all__ = [
    "AdmissionStatus",
    "FacetRegistryEntry",
    "ProfileRow",
    "ValidatorBinding",
    "ValidatorOrigin",
    "ValidatorScope",
    "method_table",
    "profile_inventory",
    "profile_row",
    "summarise_inventory",
    "validator_registry",
]
