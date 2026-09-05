# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical model identity registry and catalogue count membership

"""Canonical identity of every catalogue model and the counts derived from it.

The model registry (``neurons.models._CLASS_TO_MODULE``), the family taxonomy,
the schema alias table and the descriptor corpus each describe one facet of a
model. None of them states *what kind of identity* a class is, whether it
counts in the public source catalogue, which schema profiles belong to it, or
which evidence gates are still open. This module binds those facets into one
reviewable record per class and derives every catalogue number from it, so a
public count is a computation over the registry rather than a typed figure.

Identity kinds:

``source-literature``
    A published model with a source locator (DOI, URL or citation). Counts in
    the source catalogue.
``project-original``
    A model designed by the project without an external paper. Counts in the
    source catalogue; its source is the project specification.
``sc-compatibility``
    A preserved project recurrence retained under an explicit ``SC`` identity
    when a source correction displaced it. Count-neutral.
``api-alias``
    An import-compatible historical class name resolving to another identity.
    Count-neutral and never a catalogue row.

Ambiguous or missing joins raise :class:`ModelIdentityError` instead of being
dropped: an ``SC`` identity carrying a literature DOI, a schema stem that joins
no registered class or more than one, a public-page binding that names an
unregistered class, or an alias whose canonical class is unknown.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.model_taxonomy import _COMPATIBILITY_ALIASES as _TAXONOMY_ALIASES
from sc_neurocore.neurons.model_taxonomy import model_family
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.neurons.schema_module_aliases import class_for_schema, module_for_schema

IdentityKind = Literal["source-literature", "project-original", "sc-compatibility", "api-alias"]
SourceBasis = Literal["doi", "url", "citation-unlocated", "project-specification", "alias"]
PublicStatus = Literal[
    "polyglot-complete",
    "runtime-validated",
    "compatibility-runtime",
    "remaining",
    "unlisted",
    "alias",
]
Revalidation = Literal["receipt-bound", "not-revalidated", "not-completed"]

SCHEMA_DIR = Path(__file__).resolve().parent / "model_schemas"
REQUIRED_BACKENDS: tuple[str, ...] = ("python", "rust", "julia", "go", "mojo")
_RECEIPT_MARKER = "neurons/reference_receipts/"


class ModelIdentityError(ValueError):
    """Raised when a catalogue identity cannot be resolved unambiguously."""


@dataclass(frozen=True, slots=True)
class SchemaProfile:
    """One schema-DSL profile bound to a model identity.

    Parameters
    ----------
    stem:
        Schema file stem under ``neurons/model_schemas``.
    basis:
        ``"alias-table"`` when :mod:`schema_module_aliases` names the class,
        ``"module-stem"`` when the stem resolves through the model module.
    """

    stem: str
    basis: Literal["alias-table", "module-stem"]


@dataclass(frozen=True, slots=True)
class SourceLocator:
    """Where a model's defining source lives.

    Parameters
    ----------
    basis:
        Which locator establishes the identity.
    doi, url, paper_title, authors, year:
        Provenance fields copied from the descriptor.
    doi_is_translation:
        ``True`` when the DOI points at a translation of the primary source.
    """

    basis: SourceBasis
    doi: str = ""
    url: str = ""
    paper_title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi_is_translation: bool = False


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Canonical identity record of one catalogue class.

    Parameters
    ----------
    class_name:
        Public Python class name.
    module:
        Module stem under ``neurons/models`` (or the alias re-export module).
    kind:
        Identity kind; see the module docstring.
    counts_in_source_catalogue:
        ``True`` only for ``source-literature`` and ``project-original``.
    family, category:
        Curated taxonomy family and category slug (empty for aliases).
    canonical_class:
        The identity an alias resolves to; the class itself otherwise.
    aliases:
        Historical import names resolving to this identity.
    schema_profiles:
        Schema-DSL profiles bound to this identity.
    source:
        Source locator.
    public_status:
        Row status on the public fidelity page.
    public_label:
        Row label on the public fidelity page, empty when unlisted.
    revalidation:
        For strict-promoted identities: whether the promotion is bound to an
        independent source receipt.
    missing_gates:
        Evidence gates the descriptor does not yet claim.
    """

    class_name: str
    module: str
    kind: IdentityKind
    counts_in_source_catalogue: bool
    family: str
    category: str
    canonical_class: str
    aliases: tuple[str, ...]
    schema_profiles: tuple[SchemaProfile, ...]
    source: SourceLocator
    public_status: PublicStatus
    public_label: str
    revalidation: Revalidation
    missing_gates: tuple[str, ...]

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-compatible projection of the record.

        Returns
        -------
        dict[str, object]
            Stable field order suitable for generated ledgers.
        """
        return {
            "class_name": self.class_name,
            "module": self.module,
            "kind": self.kind,
            "counts_in_source_catalogue": self.counts_in_source_catalogue,
            "family": self.family,
            "category": self.category,
            "canonical_class": self.canonical_class,
            "aliases": list(self.aliases),
            "schema_profiles": [{"stem": p.stem, "basis": p.basis} for p in self.schema_profiles],
            "source": {
                "basis": self.source.basis,
                "doi": self.source.doi,
                "url": self.source.url,
                "paper_title": self.source.paper_title,
                "authors": list(self.source.authors),
                "year": self.source.year,
                "doi_is_translation": self.source.doi_is_translation,
            },
            "public_status": self.public_status,
            "public_label": self.public_label,
            "revalidation": self.revalidation,
            "missing_gates": list(self.missing_gates),
        }


@dataclass(frozen=True, slots=True)
class NetworkIdentity:
    """A network-level identity that is distinct from any cell component.

    Parameters
    ----------
    class_name:
        Public network class name.
    module:
        Dotted module path of the class.
    kind:
        Identity kind (``sc-compatibility`` for retained project networks).
    cell_identity:
        The neuron identity the network is built from.
    """

    class_name: str
    module: str
    kind: IdentityKind
    cell_identity: str


@dataclass(frozen=True, slots=True)
class CatalogueCounts:
    """Every catalogue number derived from the identity registry.

    Parameters
    ----------
    registered:
        Registered model classes (aliases excluded).
    source_catalogue:
        Identities that count in the public source catalogue.
    source_literature, project_original, sc_compatibility, api_aliases:
        Identity kind totals.
    network_identities:
        Registered network-level identities.
    polyglot_complete_source, polyglot_complete_sc:
        Strict-promoted rows on the public page, split by count membership.
    runtime_validated, compatibility_runtime:
        Rows in the two non-promoted public tables.
    remaining_source:
        Source-catalogue identities not strict-promoted.
    receipt_bound_complete, not_revalidated_complete:
        Strict-promoted source identities with and without an independent
        source receipt.
    schema_profiles:
        Schema-DSL stems bound to an identity.
    """

    registered: int
    source_catalogue: int
    source_literature: int
    project_original: int
    sc_compatibility: int
    api_aliases: int
    network_identities: int
    polyglot_complete_source: int
    polyglot_complete_sc: int
    runtime_validated: int
    compatibility_runtime: int
    remaining_source: int
    receipt_bound_complete: int
    not_revalidated_complete: int
    schema_profiles: int

    def to_public_dict(self) -> dict[str, int]:
        """Return the counts as a plain mapping.

        Returns
        -------
        dict[str, int]
            Field name to count.
        """
        return {
            "registered": self.registered,
            "source_catalogue": self.source_catalogue,
            "source_literature": self.source_literature,
            "project_original": self.project_original,
            "sc_compatibility": self.sc_compatibility,
            "api_aliases": self.api_aliases,
            "network_identities": self.network_identities,
            "polyglot_complete_source": self.polyglot_complete_source,
            "polyglot_complete_sc": self.polyglot_complete_sc,
            "runtime_validated": self.runtime_validated,
            "compatibility_runtime": self.compatibility_runtime,
            "remaining_source": self.remaining_source,
            "receipt_bound_complete": self.receipt_bound_complete,
            "not_revalidated_complete": self.not_revalidated_complete,
            "schema_profiles": self.schema_profiles,
        }


COUNT_DEFINITION = (
    "The source catalogue is every registered model class whose identity kind is "
    "source-literature or project-original. SC compatibility identities (class names "
    "beginning with SC) and import aliases are count-neutral. The strict "
    "polyglot-complete figure counts public fidelity rows bound to a source-catalogue "
    "identity; the remaining figure is the source catalogue minus that count."
)

# Public fidelity page rows (docs/api/model_fidelity_status.md) bound to registered
# classes. The page is prose; this table is the machine-readable binding that lets
# the public count be derived and checked. Labels must match the page row text.
_PUBLIC_FIDELITY_ROWS: dict[str, tuple[str, PublicStatus]] = {
    "WangBuzsakiNeuron": ("Wang-Buzsaki", "polyglot-complete"),
    "FitzHughNagumoNeuron": ("FitzHugh-Nagumo", "polyglot-complete"),
    "MorrisLecarNeuron": ("Morris-Lecar", "polyglot-complete"),
    "ConnorStevensNeuron": ("Connor-Stevens", "polyglot-complete"),
    "HodgkinHuxleyNeuron": ("Hodgkin-Huxley", "polyglot-complete"),
    "AdExNeuron": ("AdEx", "polyglot-complete"),
    "ExpIFNeuron": ("ExpIF", "polyglot-complete"),
    "LapicqueNeuron": ("Lapicque 1907 polarization threshold", "polyglot-complete"),
    "PerfectIntegratorNeuron": ("Perfect Integrator", "polyglot-complete"),
    "QuadraticIFNeuron": ("Quadratic IF", "polyglot-complete"),
    "ThetaNeuron": ("Theta", "polyglot-complete"),
    "DPINeuron": ("DPI", "polyglot-complete"),
    "COBALIFNeuron": ("COBA LIF", "polyglot-complete"),
    "EscapeRateNeuron": ("Escape Rate", "polyglot-complete"),
    "PoissonNeuron": ("Poisson", "polyglot-complete"),
    "IntegerQIFNeuron": ("IQIF", "polyglot-complete"),
    "McCullochPittsNeuron": ("McCulloch-Pitts", "polyglot-complete"),
    "SigmoidRateNeuron": ("Sigmoid Rate", "polyglot-complete"),
    "ThresholdLinearRateNeuron": ("Threshold-linear Rate", "polyglot-complete"),
    "WilsonCowanUnit": ("Wilson-Cowan", "polyglot-complete"),
    "JansenRitUnit": ("Jansen–Rit", "polyglot-complete"),
    "ErmentroutKopellPopulation": ("Montbrió–Pazó–Roxin", "polyglot-complete"),
    "ResonateAndFireNeuron": ("Resonate-and-Fire", "polyglot-complete"),
    "AlphaNeuron": ("Alpha-Synapse LIF", "polyglot-complete"),
    "AdaptiveThresholdIFNeuron": ("Adaptive-Threshold IF", "polyglot-complete"),
    "WongWangUnit": ("Wong-Wang", "polyglot-complete"),
    "AmariNeuralField": ("Amari neural field", "polyglot-complete"),
    "BrunelWangNeuron": ("Brunel-Wang pyramidal cell", "polyglot-complete"),
    "CompteWMNeuron": ("Compte working-memory pyramidal cell", "polyglot-complete"),
    "MATNeuron": ("Kobayashi MAT*", "polyglot-complete"),
    "NonResettingLIFNeuron": ("Kobayashi MAT(1)", "polyglot-complete"),
    "SigmaDeltaNeuron": ("Yoon asynchronous pulse sigma-delta", "polyglot-complete"),
    "EnergyLIFNeuron": ("Fardet-Levina energy-based LIF", "polyglot-complete"),
    "McKeanNeuron": ("McKean", "polyglot-complete"),
    "SCTriangularMcKeanNeuron": ("SC triangular McKean-like recurrence", "polyglot-complete"),
    "BendaHerzNeuron": ("Benda-Herz universal adaptation", "polyglot-complete"),
    "SCStochasticRateAdaptationNeuron": ("SC stochastic rate adaptation", "polyglot-complete"),
    "BertramPhantomBurster": ("Bertram phantom burster", "polyglot-complete"),
    "HillTononiNeuron": ("Hill–Tononi cortical waking neuron", "polyglot-complete"),
    "ButeraRespiratoryNeuron": ("Butera Model 1 respiratory pacemaker", "polyglot-complete"),
    "LarterBreakspearNeuron": ("Larter–Breakspear cortical neural mass", "polyglot-complete"),
    "NMDANeuron": ("Wang NMDA-autapse pyramidal neuron", "polyglot-complete"),
    "SCWBNMDAMagnesiumBlockNeuron": (
        "SC WB plus NMDA magnesium-block recurrence",
        "polyglot-complete",
    ),
    "HindmarshRoseNeuron": ("Hindmarsh-Rose", "polyglot-complete"),
    "FitzHughRinzelNeuron": ("FitzHugh-Rinzel", "polyglot-complete"),
    "PernarowskiNeuron": ("Pernarowski", "polyglot-complete"),
    "TermanWangOscillator": ("Terman-Wang", "polyglot-complete"),
    "WilsonHRNeuron": ("Wilson-HR", "polyglot-complete"),
    "RulkovMapNeuron": ("Rulkov 2002 map", "polyglot-complete"),
    "SCUpwardCrossingRulkovMapNeuron": ("SC upward-crossing Rulkov map", "polyglot-complete"),
    "GLIFNeuron": ("GLIF5", "polyglot-complete"),
    "SCFourStateGLIFNeuron": ("SC four-state GLIF recurrence", "polyglot-complete"),
    "MihalasNieburNeuron": ("Mihalas-Niebur", "polyglot-complete"),
    "SCScaledResetAdaptiveIFNeuron": ("SC scaled-reset adaptive IF", "polyglot-complete"),
    "MedvedevMapNeuron": ("Medvedev map", "polyglot-complete"),
    "CazellesMapNeuron": ("Cazelles map", "polyglot-complete"),
    "SCClippedLogisticBurstingMapNeuron": (
        "SC clipped-logistic bursting map",
        "polyglot-complete",
    ),
    "ChialvoMapNeuron": ("Chialvo map", "polyglot-complete"),
    "AiharaMapNeuron": ("Aihara map", "polyglot-complete"),
    "NagumoSatoMapNeuron": ("Nagumo–Sato map", "polyglot-complete"),
    "SCAdaptiveThresholdMapNeuron": ("SC adaptive-threshold map", "polyglot-complete"),
    "SCChaoticMapNeuron": ("SC chaotic map", "polyglot-complete"),
    "CourageNekorkinMapNeuron": ("Courbage-Nekorkin-Vdovin map", "polyglot-complete"),
    "SCClippedRationalRecoveryMapNeuron": (
        "SC clipped rational-recovery map",
        "polyglot-complete",
    ),
    "Izhikevich2007Neuron": ("Izhikevich 2007", "polyglot-complete"),
    "IbarzTanakaMapNeuron": ("Ibarz-Tanaka analysis profile", "polyglot-complete"),
    "ErmentroutKopellMapNeuron": ("Ermentrout-Kopell", "polyglot-complete"),
    "SKNeuron": ("SK neuron", "runtime-validated"),
    "TTypeCaNeuron": ("T-type calcium neuron", "runtime-validated"),
    "GLMNeuron": ("GLM neuron", "runtime-validated"),
    "SCThreeStatePhantomBurster": ("SC three-state phantom", "compatibility-runtime"),
}

# Network-level identities distinct from their cell components.
NETWORK_IDENTITIES: tuple[NetworkIdentity, ...] = (
    NetworkIdentity(
        class_name="SCCompteWMNetwork",
        module="sc_neurocore.network.sc_compte_wm_network",
        kind="sc-compatibility",
        cell_identity="CompteWMNeuron",
    ),
)


def _schema_stems() -> tuple[str, ...]:
    if not SCHEMA_DIR.is_dir():
        return ()
    return tuple(
        sorted({path.stem for path in SCHEMA_DIR.iterdir() if path.suffix in {".toml", ".json"}})
    )


def _classes_by_module() -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for class_name, module in _CLASS_TO_MODULE.items():
        grouped.setdefault(module, []).append(class_name)
    return {module: tuple(sorted(names)) for module, names in grouped.items()}


def _bind_schema_profiles() -> dict[str, tuple[SchemaProfile, ...]]:
    """Return every schema stem bound to exactly one registered class."""
    by_module = _classes_by_module()
    bound: dict[str, list[SchemaProfile]] = {}
    for stem in _schema_stems():
        class_name = class_for_schema(stem)
        basis: Literal["alias-table", "module-stem"] = "alias-table"
        if class_name is None:
            candidates = by_module.get(module_for_schema(stem), ())
            if len(candidates) == 1:
                class_name = candidates[0]
                basis = "module-stem"
            elif len(candidates) > 1:
                raise ModelIdentityError(
                    f"schema stem {stem!r} joins more than one registered class "
                    f"{list(candidates)}; add an explicit alias-table entry"
                )
            else:
                raise ModelIdentityError(
                    f"schema stem {stem!r} joins no registered class; add an alias-table entry"
                )
        if class_name not in _CLASS_TO_MODULE:
            raise ModelIdentityError(
                f"schema stem {stem!r} names unregistered class {class_name!r}"
            )
        bound.setdefault(class_name, []).append(SchemaProfile(stem=stem, basis=basis))
    return {name: tuple(profiles) for name, profiles in bound.items()}


def _provenance(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if payload is None:
        return {}
    section = payload.get("provenance")
    return section if isinstance(section, Mapping) else {}


def _str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return ()


def _source_locator(class_name: str, payload: Mapping[str, Any] | None) -> SourceLocator:
    provenance = _provenance(payload)
    doi = str(provenance.get("doi", "") or "")
    url = str(provenance.get("url", "") or "")
    paper_title = str(provenance.get("paper_title", "") or "")
    authors = _str_tuple(provenance.get("authors"))
    raw_year = provenance.get("year")
    year = raw_year if isinstance(raw_year, int) and not isinstance(raw_year, bool) else None
    translation = bool(provenance.get("doi_is_translation", False))
    if class_name.startswith("SC"):
        if doi:
            raise ModelIdentityError(
                f"{class_name} is an SC compatibility identity but carries literature DOI {doi!r}"
            )
        basis: SourceBasis = "project-specification"
    elif doi:
        basis = "doi"
    elif url:
        basis = "url"
    elif paper_title or any(not _is_project_author(author) for author in authors):
        basis = "citation-unlocated"
    else:
        basis = "project-specification"
    return SourceLocator(
        basis=basis,
        doi=doi,
        url=url,
        paper_title=paper_title,
        authors=authors,
        year=year,
        doi_is_translation=translation,
    )


def _is_project_author(author: str) -> bool:
    lowered = author.lower()
    return "sc-neurocore" in lowered or "anulum" in lowered or "arcane sapience" in lowered


def _identity_kind(class_name: str, source: SourceLocator) -> IdentityKind:
    if class_name.startswith("SC"):
        return "sc-compatibility"
    if source.basis == "project-specification":
        return "project-original"
    return "source-literature"


def _missing_gates(
    payload: Mapping[str, Any] | None,
    profiles: tuple[SchemaProfile, ...],
    source: SourceLocator,
    kind: IdentityKind,
) -> tuple[str, ...]:
    gates: list[str] = []
    if kind == "source-literature" and source.basis == "citation-unlocated":
        gates.append("source-locator")
    if not profiles:
        gates.append("schema-dsl-profile")
    if payload is None:
        gates.append("descriptor")
        return tuple(gates)
    backends = payload.get("backends")
    backend_map = backends if isinstance(backends, Mapping) else {}
    for name in REQUIRED_BACKENDS:
        entry = backend_map.get(name)
        status = entry.get("status") if isinstance(entry, Mapping) else entry
        if status != "implemented":
            gates.append(f"backend:{name}")
    validation = payload.get("validation")
    validation_map = validation if isinstance(validation, Mapping) else {}
    if not bool(validation_map.get("dynamics_faithful", False)):
        gates.append("independent-source-validation")
    silicon = payload.get("silicon")
    silicon_map = silicon if isinstance(silicon, Mapping) else {}
    for field_name, gate in (
        ("compiles", "rtl-compile"),
        ("cosim_validated", "cosim"),
        ("synthesised", "synthesis"),
        ("timing_closed", "timing"),
        ("formally_equivalent", "formal-equivalence"),
        ("ppa_signed", "ppa"),
    ):
        if not bool(silicon_map.get(field_name, False)):
            gates.append(gate)
    return tuple(gates)


def _revalidation(payload: Mapping[str, Any] | None, status: PublicStatus) -> Revalidation:
    if status != "polyglot-complete":
        return "not-completed"
    if payload is None:
        return "not-revalidated"
    reproducibility = payload.get("reproducibility")
    repro_map = reproducibility if isinstance(reproducibility, Mapping) else {}
    reference = str(repro_map.get("reference_config", "") or "")
    validation = payload.get("validation")
    validation_map = validation if isinstance(validation, Mapping) else {}
    faithful = bool(validation_map.get("dynamics_faithful", False))
    if _RECEIPT_MARKER in reference and faithful:
        return "receipt-bound"
    return "not-revalidated"


def _validate_public_rows() -> None:
    labels: dict[str, str] = {}
    for class_name, (label, _status) in _PUBLIC_FIDELITY_ROWS.items():
        if class_name not in _CLASS_TO_MODULE:
            raise ModelIdentityError(
                f"public fidelity row {label!r} names unregistered class {class_name!r}"
            )
        if label in labels:
            raise ModelIdentityError(
                f"public fidelity label {label!r} is bound to both {labels[label]} and {class_name}"
            )
        labels[label] = class_name


def _validate_aliases() -> dict[str, tuple[str, ...]]:
    reverse: dict[str, list[str]] = {}
    for alias, canonical in _TAXONOMY_ALIASES.items():
        if canonical not in _CLASS_TO_MODULE:
            raise ModelIdentityError(
                f"alias {alias!r} resolves to unregistered class {canonical!r}"
            )
        if alias in _CLASS_TO_MODULE:
            raise ModelIdentityError(f"alias {alias!r} is also a registered catalogue class")
        reverse.setdefault(canonical, []).append(alias)
    return {name: tuple(sorted(aliases)) for name, aliases in reverse.items()}


@lru_cache(maxsize=1)
def identity_registry() -> dict[str, ModelIdentity]:
    """Return the canonical identity record for every registered class and alias.

    Returns
    -------
    dict[str, ModelIdentity]
        Class name (including aliases) to identity record, sorted by name.

    Raises
    ------
    ModelIdentityError
        If any join is ambiguous or names an unknown class.
    """
    _validate_public_rows()
    aliases_of = _validate_aliases()
    profiles_of = _bind_schema_profiles()
    records: dict[str, ModelIdentity] = {}
    for class_name in sorted(_CLASS_TO_MODULE):
        payload = load_descriptor_payload(class_name)
        source = _source_locator(class_name, payload)
        kind = _identity_kind(class_name, source)
        family_category = model_family(class_name) or ("", "")
        profiles = profiles_of.get(class_name, ())
        label, status = _PUBLIC_FIDELITY_ROWS.get(class_name, ("", "remaining"))
        if status == "remaining" and kind == "sc-compatibility":
            status = "unlisted"
        records[class_name] = ModelIdentity(
            class_name=class_name,
            module=_CLASS_TO_MODULE[class_name],
            kind=kind,
            counts_in_source_catalogue=kind in {"source-literature", "project-original"},
            family=family_category[0],
            category=family_category[1],
            canonical_class=class_name,
            aliases=aliases_of.get(class_name, ()),
            schema_profiles=profiles,
            source=source,
            public_status=status,
            public_label=label,
            revalidation=_revalidation(payload, status),
            missing_gates=_missing_gates(payload, profiles, source, kind),
        )
    for alias, canonical in sorted(_TAXONOMY_ALIASES.items()):
        target = records[canonical]
        records[alias] = ModelIdentity(
            class_name=alias,
            module=target.module,
            kind="api-alias",
            counts_in_source_catalogue=False,
            family=target.family,
            category=target.category,
            canonical_class=canonical,
            aliases=(),
            schema_profiles=(),
            source=SourceLocator(basis="alias"),
            public_status="alias",
            public_label="",
            revalidation="not-completed",
            missing_gates=(),
        )
    return dict(sorted(records.items()))


def resolve_identity(class_name: str) -> ModelIdentity:
    """Return the canonical identity for a class or alias name.

    Parameters
    ----------
    class_name:
        Registered class name or historical alias.

    Returns
    -------
    ModelIdentity
        The record of the canonical identity the name resolves to.

    Raises
    ------
    ModelIdentityError
        If the name is neither registered nor an alias.
    """
    registry = identity_registry()
    record = registry.get(class_name)
    if record is None:
        raise ModelIdentityError(f"{class_name!r} is not a registered catalogue identity or alias")
    return registry[record.canonical_class]


def iter_source_catalogue() -> Iterator[ModelIdentity]:
    """Yield identities that count in the public source catalogue."""
    for record in identity_registry().values():
        if record.counts_in_source_catalogue:
            yield record


def public_fidelity_bindings() -> dict[str, tuple[str, PublicStatus]]:
    """Return the public page label and status bound to each listed class.

    Returns
    -------
    dict[str, tuple[str, PublicStatus]]
        Class name to ``(row label, status)``.
    """
    return dict(_PUBLIC_FIDELITY_ROWS)


def catalogue_counts() -> CatalogueCounts:
    """Derive every catalogue number from the identity registry.

    Returns
    -------
    CatalogueCounts
        Counts computed over :func:`identity_registry`.
    """
    registry = identity_registry()
    classes = [record for record in registry.values() if record.kind != "api-alias"]
    source = [record for record in classes if record.counts_in_source_catalogue]
    complete_source = [record for record in source if record.public_status == "polyglot-complete"]
    complete_sc = [
        record
        for record in classes
        if record.public_status == "polyglot-complete" and not record.counts_in_source_catalogue
    ]
    return CatalogueCounts(
        registered=len(classes),
        source_catalogue=len(source),
        source_literature=sum(1 for record in classes if record.kind == "source-literature"),
        project_original=sum(1 for record in classes if record.kind == "project-original"),
        sc_compatibility=sum(1 for record in classes if record.kind == "sc-compatibility"),
        api_aliases=sum(1 for record in registry.values() if record.kind == "api-alias"),
        network_identities=len(NETWORK_IDENTITIES),
        polyglot_complete_source=len(complete_source),
        polyglot_complete_sc=len(complete_sc),
        runtime_validated=sum(
            1 for record in classes if record.public_status == "runtime-validated"
        ),
        compatibility_runtime=sum(
            1 for record in classes if record.public_status == "compatibility-runtime"
        ),
        remaining_source=len(source) - len(complete_source),
        receipt_bound_complete=sum(
            1 for record in complete_source if record.revalidation == "receipt-bound"
        ),
        not_revalidated_complete=sum(
            1 for record in complete_source if record.revalidation == "not-revalidated"
        ),
        schema_profiles=sum(len(record.schema_profiles) for record in classes),
    )


__all__ = [
    "COUNT_DEFINITION",
    "NETWORK_IDENTITIES",
    "REQUIRED_BACKENDS",
    "CatalogueCounts",
    "IdentityKind",
    "ModelIdentity",
    "ModelIdentityError",
    "NetworkIdentity",
    "PublicStatus",
    "Revalidation",
    "SchemaProfile",
    "SourceBasis",
    "SourceLocator",
    "catalogue_counts",
    "identity_registry",
    "iter_source_catalogue",
    "public_fidelity_bindings",
    "resolve_identity",
]
