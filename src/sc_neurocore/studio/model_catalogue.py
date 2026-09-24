# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio model catalogue metadata

"""Descriptor-driven model catalogue for Studio discovery UX."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from sc_neurocore.neurons.descriptor_tiers import SILICON_RUNGS, completeness_tiers, is_perfect
from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_identity import identity_registry
from sc_neurocore.neurons.model_descriptor import (
    ModelDescriptor,
    descriptor_completeness_tier,
)
from sc_neurocore.neurons.model_profile import resolve_profile
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.neurons.schema_module_aliases import schema_for_module
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema
from sc_neurocore.studio.model_numeric_contracts import (
    bit_true_mirrored,
    studio_numeric_contracts,
)
from sc_neurocore.studio.readiness_seal import (
    SOURCE_RECEIPTS,
    build_seal,
    checkout_available,
    sealed_detail,
)
from sc_neurocore.studio.model_introspection import (
    _categorize,
    _classify_fields,
    _extract_dt,
    _load_class,
)


def _evidence_kind(tier: int) -> str:
    """Map a completeness tier to the SCPN-Studio evidence modality."""
    if tier >= 3:
        return "measured"
    if tier == 2:
        return "curated"
    return ""


#: Per-entry metadata health. ``available`` is a declared descriptor;
#: ``unavailable`` is a real model whose descriptor is absent, described by code
#: introspection instead; ``invalid`` is a model whose metadata could not be
#: read at all. An ``invalid`` entry is still listed — a catalogue identity that
#: vanishes on a metadata fault reports a smaller success count rather than a
#: fault, and takes every list-scoped consumer's scope down with it.
METADATA_STATE_AVAILABLE = "available"
METADATA_STATE_UNAVAILABLE = "unavailable"
METADATA_STATE_INVALID = "invalid"

_models_cache: list[dict[str, Any]] | None = None
logger = logging.getLogger(__name__)


class ModelMetadataError(RuntimeError):
    """Raised when Studio model metadata loading fails for a known model."""


def _identity_fields(name: str) -> dict[str, Any]:
    """Return one model's catalogue-identity classification.

    The corpus is not one population: of 185 registered identities, 133 come
    from published literature, 27 are project-original, 25 are SC-compatibility
    identities and one is an API alias. A browser that shows only a total
    invites a reader to count all of them as literature models, which the
    backlog names directly — "class aliases do not inflate literature count".

    Parameters
    ----------
    name : str
        Registered model identity.

    Returns
    -------
    dict
        ``identity_kind``, whether it ``counts_in_source_catalogue``, the
        ``public_label`` it is published under, and its ``aliases``. Empty
        values when the registry does not hold the name, which is stated rather
        than guessed.
    """
    identity = identity_registry().get(name)
    if identity is None:
        return {
            "identity_kind": "",
            "counts_in_source_catalogue": False,
            "public_label": "",
            "aliases": [],
        }
    return {
        "identity_kind": identity.kind,
        "counts_in_source_catalogue": identity.counts_in_source_catalogue,
        "public_label": identity.public_label,
        "aliases": list(identity.aliases),
    }


def _provenance_summary(descriptor: ModelDescriptor) -> dict[str, Any] | None:
    """Return a path-free provenance summary, or ``None`` when uncited."""
    prov = descriptor.provenance
    if not (prov.authors or prov.year or prov.doi):
        return None
    return {
        "authors": list(prov.authors),
        "year": prov.year,
        "doi": prov.doi,
        "paper_title": prov.paper_title,
        "url": prov.url,
        "citeable": prov.is_citeable,
    }


def _descriptor_summary(descriptor: ModelDescriptor) -> dict[str, Any]:
    """Build a catalogue list entry from a declared descriptor."""
    tier = descriptor_completeness_tier(descriptor)
    tiers = completeness_tiers(descriptor)
    verified = _verified_summary(descriptor.class_name)
    return {
        "name": descriptor.class_name,
        "module": descriptor.module,
        **_identity_fields(descriptor.class_name),
        "metadata_state": METADATA_STATE_AVAILABLE,
        "metadata_error": None,
        "tier": tier,
        "evidence_kind": _evidence_kind(tier),
        "science_tier": tiers.science,
        "science_label": tiers.science_label,
        "silicon_tier": tiers.silicon,
        "silicon_label": tiers.silicon_label,
        **verified,
        # The same rule as the detail view's judgement, applied to the verified
        # tiers, so a browser can filter on proven readiness without fetching
        # every model's detail.
        "is_perfect_verified": _verified_perfect(
            verified["verified_science_tier"],
            verified["verified_silicon_tier"],
            descriptor.silicon.target_tier,
        ),
        "validation_metric": descriptor.validation.metric,
        "integration_method": descriptor.integration_method,
        "terminal_silicon_tier": descriptor.silicon.target_tier,
        "terminal_reason": descriptor.silicon.terminal_reason,
        # ``category`` carries the family display name so existing clients group
        # by the curated family; the fine slug is exposed separately.
        "category": descriptor.family,
        "category_slug": descriptor.category,
        "category_source": "declared",
        "family": descriptor.family,
        "maturity": descriptor.maturity,
        "biophysical_detail": descriptor.biophysical_detail,
        "n_state_vars": len(descriptor.state),
        "n_params": len(descriptor.parameters),
        "state_var_names": [s.name for s in descriptor.state],
        "dt": descriptor.dt,
        "description": descriptor.summary,
        "intended_use": list(descriptor.intended_use),
        "hardware_fit": list(descriptor.hardware_fit),
        "behavior_tags": list(descriptor.behavior_tags),
        "provenance": _provenance_summary(descriptor),
    }


def _descriptor_detail(descriptor: ModelDescriptor) -> dict[str, Any]:
    """Build a full catalogue detail view from a declared descriptor."""
    detail = _descriptor_summary(descriptor)
    detail.update(
        {
            "docstring": descriptor.summary,
            "display_name": descriptor.display_name,
            "state_vars": [
                {"name": s.name, "default": s.init, "unit": s.unit, "meaning": s.meaning}
                for s in descriptor.state
            ],
            "params": [
                {
                    "name": p.name,
                    "default": p.default,
                    "unit": p.unit,
                    "range": list(p.value_range) if p.value_range else None,
                    "biological_range": (list(p.biological_range) if p.biological_range else None),
                    "meaning": p.meaning,
                }
                for p in descriptor.parameters
            ],
            "dynamics": dict(descriptor.dynamics),
            "backends": [
                {"name": b.name, "status": b.status, "parity": b.parity}
                for b in descriptor.backends
            ],
            "reproducibility": {
                "reference_config": descriptor.reproducibility.reference_config,
                "golden_trace_sha256": descriptor.reproducibility.golden_trace_sha256,
                "golden_trace_sha256_variants": list(
                    descriptor.reproducibility.golden_trace_sha256_variants
                ),
                "reproducible": descriptor.reproducibility.is_reproducible,
            },
            "readiness": _readiness_detail(descriptor),
            "documentation_slug": descriptor.documentation_slug,
            "compile_configuration": _compile_configuration(descriptor),
            "profile_contract": _profile_contract(descriptor),
        }
    )
    return detail


def _canonical_schema(descriptor: ModelDescriptor) -> tuple[str, dict[str, Any]] | None:
    """Return the class's canonical schema stem and document, if one is bundled."""
    from sc_neurocore.neurons.model_identity import ModelIdentityError, schema_for_class

    try:
        schema_name = schema_for_class(descriptor.class_name)
    except ModelIdentityError:
        schema_name = schema_for_module(descriptor.module.rsplit(".", 1)[-1])
    try:
        return schema_name, load_schema(schema_name)
    except (FileNotFoundError, ValueError):
        return None


def _profile_contract(descriptor: ModelDescriptor) -> dict[str, Any] | None:
    """Return the canonical schema's resolved model profile, if one is bundled.

    The profile separates the scientific model from its numerical realisation
    and lowering profile (:mod:`sc_neurocore.neurons.model_profile`); Studio
    shows it next to the descriptor so a user sees which state is biological,
    which parameters are implementation choices, how sub-steps are meant and
    which methods may be selected without leaving the profile's family.
    """
    canonical = _canonical_schema(descriptor)
    if canonical is None:
        return None
    schema_name, schema = canonical
    return resolve_profile(schema, stem=schema_name).to_public_dict()


def _compile_configuration(descriptor: ModelDescriptor) -> dict[str, Any] | None:
    """Return the canonical schema-backed Studio compile choices, if available.

    Declared ``extensions.integrator_options`` are admitted only inside the
    profile's numerical family: a published map never offers an ODE integrator
    and an ODE never offers ``map``. A schema whose profile is contradictory or
    a descriptive record has no compile configuration.

    Q-formats are offered only where the schema's neuron is representable (see
    :mod:`sc_neurocore.studio.model_numeric_contracts`), smallest word first,
    and the first is the default; ``numeric_contracts`` states what the RTL
    holds at every candidate format, including the refused ones and why. A
    neuron no candidate can hold has no default and no offered format.
    ``cosim_integrators`` keeps only integrators whose neuron a generated
    bit-true C kernel mirrors, since the co-simulation compares against it.
    """
    canonical = _canonical_schema(descriptor)
    if canonical is None:
        return None
    schema_name, schema = canonical
    profile = resolve_profile(schema, stem=schema_name)
    if profile.problems or not profile.is_executable:
        return None
    default_integrator = profile.numerical.method
    if default_integrator not in SUPPORTED_METHODS:
        return None
    admissible = set(profile.numerical.admissible_methods)
    extensions = schema.get("extensions", {})
    declared = extensions.get("integrator_options", [default_integrator])
    integrators = [
        str(value)
        for value in declared
        if isinstance(value, str) and value in SUPPORTED_METHODS and value in admissible
    ]
    if default_integrator not in integrators:
        integrators.insert(0, default_integrator)
    contracts = studio_numeric_contracts(UniversalNeuron.from_schema(schema_name))
    q_formats = [label for label, contract in contracts.items() if contract.representable]
    return {
        "schema_name": schema_name,
        "default_integrator": default_integrator,
        "integrators": list(dict.fromkeys(integrators)),
        "cosim_integrators": [
            integrator
            for integrator in dict.fromkeys(integrators)
            if integrator in profile.lowering.cosim_methods
            and q_formats
            and bit_true_mirrored(schema_name, integrator, q_formats[0])
        ],
        "default_q_format": q_formats[0] if q_formats else None,
        "q_formats": q_formats,
        "numeric_contracts": {
            label: contract.to_public_dict() for label, contract in contracts.items()
        },
    }


def _selected_profile(class_name: str) -> str | None:
    """Return the concrete profile Studio verifies a class under.

    Receipts are keyed by profile and a multi-profile identity receives no
    class-wide verified claim without a selection, so Studio selects the
    class's canonical schema profile (the one it compiles). A class without a
    bound schema profile is verified under its hand profile.
    """
    from sc_neurocore.neurons.model_identity import ModelIdentityError, resolve_identity

    try:
        identity = resolve_identity(class_name)
    except ModelIdentityError:
        return None
    stems = {profile.stem for profile in identity.schema_profiles}
    canonical = _canonical_schema_name(class_name)
    return canonical if canonical in stems else None


def _canonical_schema_name(class_name: str) -> str:
    from sc_neurocore.neurons.model_identity import ModelIdentityError, schema_for_class

    try:
        return schema_for_class(class_name)
    except ModelIdentityError:
        return ""


def _verified_summary(class_name: str) -> dict[str, Any]:
    """Return the verified (receipt-bound) tiers for a browse entry.

    The declared tiers above come from the descriptor's own flags; these come
    only from facet receipts, for the class's canonical profile, whose subjects
    still match the repository, so a browse entry always shows both what is
    claimed and what is proven.
    """
    from sc_neurocore.neurons.model_identity import identity_registry

    if class_name not in identity_registry():
        return {
            "verified_science_tier": 0,
            "verified_science_label": "S0",
            "verified_silicon_tier": None,
            "verified_silicon_label": "none",
            "verified_profile": None,
        }
    detail = _verified_detail(class_name)
    return {
        "verified_science_tier": detail["science_tier"],
        "verified_science_label": detail["science_label"],
        "verified_silicon_tier": detail["silicon_tier"],
        "verified_silicon_label": detail["silicon_label"],
        "verified_profile": detail["profile"],
    }


def _verified_detail(class_name: str) -> dict[str, Any]:
    """Return the per-facet verification block, re-derived or sealed.

    A checkout re-derives it from the receipts on every read (``source``
    ``receipts``). An installation cannot, because receipt subjects such as the
    validator tests are not installed, so it serves the record sealed in the
    checkout the distribution was built from (``source`` ``sealed``).
    """
    from sc_neurocore.neurons.readiness import REPO_ROOT

    if checkout_available(REPO_ROOT):
        return {**_receipt_verified_detail(class_name), "source": SOURCE_RECEIPTS}
    return sealed_detail(class_name)


def readiness_seal_payload() -> dict[str, Any]:
    """Return the verified-readiness seal of every registered catalogue model."""
    from sc_neurocore.neurons.model_identity import identity_registry

    registered = [name for name in _CLASS_TO_MODULE if name in identity_registry()]
    return build_seal(registered, _receipt_verified_detail)


def _receipt_verified_detail(class_name: str) -> dict[str, Any]:
    """Return the per-facet verification block derived from the receipts now.

    The block names the profile the receipts were read for; a receipt of
    another profile of the same class never appears here.
    """
    from sc_neurocore.neurons.readiness import verify_model

    record = verify_model(class_name, profile=_selected_profile(class_name))
    return {
        "profile": record.profile,
        "science_tier": record.verified_science,
        "science_label": record.verified_science_label,
        "silicon_tier": record.verified_silicon,
        "silicon_label": record.verified_silicon_label,
        "facets": [
            {
                "facet": facet.facet,
                "declared": facet.declared,
                "status": facet.status,
                "receipt": facet.receipt,
                "changed_subjects": list(facet.changed_subjects),
                "problems": list(facet.problems),
                "evidence": [reference.to_public_dict() for reference in facet.evidence],
            }
            for facet in record.facets
        ],
    }


def _verified_perfect(science: int, silicon: int | None, target: str | None) -> bool:
    """Return whether verified tiers meet S5 and the declared terminal silicon tier."""
    if science != 5 or silicon is None or target not in SILICON_RUNGS:
        return False
    return silicon >= SILICON_RUNGS.index(target)


def _readiness_detail(descriptor: ModelDescriptor) -> dict[str, Any]:
    """Build the auditable dual-axis readiness view for a declared descriptor.

    Surfaces the declared science (S0-S5) and silicon (H0-H5) tiers together
    with the raw evidence facets that justify them, and next to them the
    verified tiers and per-facet statuses derived from facet receipts, so a
    reviewer can see exactly why a model sits where it does, whether it meets
    its declared deployability class, and how much of the claim is bound to an
    executed, still-fresh receipt.

    ``is_perfect`` is the declared judgement; ``is_perfect_verified`` applies
    the same rule to the verified tiers, and only it may be shown as perfect.
    """
    tiers = completeness_tiers(descriptor)
    verified = _verified_detail(descriptor.class_name)
    return {
        "science_tier": tiers.science,
        "science_label": tiers.science_label,
        "silicon_tier": tiers.silicon,
        "silicon_label": tiers.silicon_label,
        "verified": verified,
        "is_perfect": is_perfect(descriptor),
        "is_perfect_verified": _verified_perfect(
            verified["science_tier"], verified["silicon_tier"], descriptor.silicon.target_tier
        ),
        "terminal_silicon_tier": descriptor.silicon.target_tier,
        "terminal_reason": descriptor.silicon.terminal_reason,
        "validation": {
            "dynamics_faithful": descriptor.validation.dynamics_faithful,
            "metric": descriptor.validation.metric,
            "operating_point": descriptor.validation.operating_point,
            "tolerance": descriptor.validation.tolerance,
            "evidence": descriptor.validation.evidence,
        },
        "silicon": {
            "compiles": descriptor.silicon.compiles,
            "cosim_validated": descriptor.silicon.cosim_validated,
            "synthesised": descriptor.silicon.synthesised,
            "timing_closed": descriptor.silicon.timing_closed,
            "formally_equivalent": descriptor.silicon.formally_equivalent,
            "ppa_signed": descriptor.silicon.ppa_signed,
            "target_device": descriptor.silicon.target_device,
            "clock_mhz": descriptor.silicon.clock_mhz,
        },
    }


def _introspected_summary(name: str) -> dict[str, Any]:
    """Fallback catalogue entry for a model with no committed descriptor."""
    cls = _load_class(name)
    state_vars, params = _classify_fields(cls)
    return {
        "name": name,
        "module": _CLASS_TO_MODULE[name],
        **_identity_fields(name),
        "metadata_state": METADATA_STATE_UNAVAILABLE,
        "metadata_error": None,
        "tier": 0,
        "evidence_kind": "",
        "science_tier": 0,
        "science_label": "S0",
        "silicon_tier": None,
        "verified_science_tier": 0,
        "verified_science_label": "S0",
        "verified_silicon_tier": None,
        "verified_silicon_label": "none",
        "verified_profile": None,
        "is_perfect_verified": False,
        "silicon_label": "none",
        "validation_metric": "none",
        "integration_method": "unknown",
        "terminal_silicon_tier": "",
        "terminal_reason": "Descriptor unavailable; no terminal silicon target declared.",
        "category": _categorize(name),
        "category_slug": "",
        "category_source": "inferred",
        "family": _categorize(name),
        "maturity": "experimental",
        "biophysical_detail": "point",
        "n_state_vars": len(state_vars),
        "n_params": len(params),
        "state_var_names": [s["name"] for s in state_vars],
        "dt": _extract_dt(cls),
        "description": (cls.__doc__ or "").strip().split("\n")[0],
        "intended_use": [],
        "hardware_fit": [],
        "behavior_tags": [],
        "provenance": None,
    }


def _unreadable_summary(name: str) -> dict[str, Any]:
    """Catalogue entry for a model whose metadata could not be read.

    Carries the same keys as a healthy entry so every consumer keeps working on
    a corpus with a fault in it, and states the fault rather than omitting the
    row.

    Parameters
    ----------
    name : str
        The registered model identity.

    Returns
    -------
    dict
        A catalogue entry with ``metadata_state`` ``"invalid"``.
    """
    return {
        "name": name,
        "module": _CLASS_TO_MODULE[name],
        **_identity_fields(name),
        "metadata_state": METADATA_STATE_INVALID,
        "metadata_error": "Model metadata unavailable",
        "tier": 0,
        "evidence_kind": "",
        "science_tier": 0,
        "science_label": "S0",
        "silicon_tier": None,
        "silicon_label": "none",
        "verified_science_tier": 0,
        "verified_science_label": "S0",
        "verified_silicon_tier": None,
        "verified_silicon_label": "none",
        "verified_profile": None,
        "is_perfect_verified": False,
        "validation_metric": "none",
        "integration_method": "unknown",
        "terminal_silicon_tier": "",
        "terminal_reason": "Model metadata could not be read; no terminal silicon target known.",
        "category": "unknown",
        "category_slug": "",
        "category_source": "unreadable",
        "family": "unknown",
        "maturity": "experimental",
        "biophysical_detail": "point",
        "n_state_vars": 0,
        "n_params": 0,
        "state_var_names": [],
        "dt": None,
        "description": "",
        "intended_use": [],
        "hardware_fit": [],
        "behavior_tags": [],
        "provenance": None,
    }


def corpus_revision(models: list[dict[str, Any]]) -> str:
    """Return a digest identifying the catalogue corpus and its health.

    Two clients holding the same revision hold the same identities in the same
    metadata states. The digest changes when a model is added or removed and
    when any entry's metadata state changes, so a client can tell a healthy
    corpus from a degraded one of the same size.

    Parameters
    ----------
    models : list of dict
        Catalogue entries as :func:`list_models` returns them.

    Returns
    -------
    str
        A 16-character hexadecimal digest.
    """
    lines = sorted(
        f"{entry['name']}:{entry['module']}:{entry['metadata_state']}" for entry in models
    )
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return digest[:16]


def list_models() -> list[dict[str, Any]]:
    """Return declared metadata for every registered neuron model.

    Each entry is built from the model's committed descriptor (family, category,
    maturity, provenance, parameter and state counts). Models without a descriptor
    fall back to code introspection with an ``inferred`` category and
    ``metadata_state`` ``"unavailable"``.

    **Every registered model is returned.** A model whose metadata cannot be read
    is reported with ``metadata_state`` ``"invalid"`` and a ``metadata_error``,
    never dropped: an omitted row shows a smaller success count instead of a
    fault, and silently narrows every consumer that derives its scope from this
    list — the runtime-state conformance matrix among them.

    Results are cached after the first call.

    Returns
    -------
    list of dict
        One entry per registered model, sorted by identity.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    result = []
    for name in sorted(_CLASS_TO_MODULE.keys()):
        try:
            descriptor = load_descriptor(name)
            entry = (
                _descriptor_summary(descriptor)
                if descriptor is not None
                else _introspected_summary(name)
            )
        except (TypeError, AttributeError, ValueError):
            logger.exception("Studio model metadata unavailable for %s", name)
            entry = _unreadable_summary(name)
        result.append(entry)
    _models_cache = result
    return result


def get_model_detail(name: str) -> dict[str, Any] | None:
    """Return the full declared metadata view for a single model."""
    if name not in _CLASS_TO_MODULE:
        return None
    try:
        descriptor = load_descriptor(name)
    except Exception as exc:
        raise ModelMetadataError(f"Failed to load Studio model descriptor for '{name}'") from exc
    if descriptor is not None:
        return _descriptor_detail(descriptor)
    try:
        cls = _load_class(name)
        state_vars, params = _classify_fields(cls)
        dt_val = _extract_dt(cls)
    except Exception as exc:
        raise ModelMetadataError(f"Failed to classify Studio model metadata for '{name}'") from exc
    return {
        **_introspected_summary(name),
        "docstring": (cls.__doc__ or "").strip().split("\n")[0],
        "state_vars": state_vars,
        "params": params,
        "dt": dt_val,
    }


def model_facets() -> dict[str, Any]:
    """Return the catalogue facet taxonomy, counts, and corpus health.

    Returns
    -------
    dict
        ``total`` registered identities, a ``corpus_revision`` digest, a
        ``metadata_states`` census, the ``invalid_models`` by name, and the
        family, maturity, behaviour and tier facets.
    """
    from collections import Counter

    models = list_models()
    family_counts: Counter[tuple[str, str]] = Counter()
    maturity_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    science_tier_counts: Counter[str] = Counter()
    silicon_tier_counts: Counter[str] = Counter()
    verified_science_counts: Counter[str] = Counter()
    verified_silicon_counts: Counter[str] = Counter()
    for model in models:
        family_counts[(str(model["family"]), str(model["category_slug"]))] += 1
        maturity_counts[str(model["maturity"])] += 1
        science_tier_counts[str(model.get("science_label", "S0"))] += 1
        silicon_tier_counts[str(model.get("silicon_label", "none"))] += 1
        verified_science_counts[str(model.get("verified_science_label", "S0"))] += 1
        verified_silicon_counts[str(model.get("verified_silicon_label", "none"))] += 1
        for tag in model.get("behavior_tags", []):
            behavior_counts[str(tag)] += 1
    families = [
        {"family": family, "category_slug": slug, "count": count}
        for (family, slug), count in sorted(family_counts.items())
    ]
    # Most-common behaviour first so the discovery UX leads with the richest filters.
    behaviors = [
        {"tag": tag, "count": count}
        for tag, count in sorted(behavior_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    metadata_states = {
        METADATA_STATE_AVAILABLE: 0,
        METADATA_STATE_UNAVAILABLE: 0,
        METADATA_STATE_INVALID: 0,
    }
    identity_kinds: Counter[str] = Counter()
    source_catalogue_total = 0
    for model in models:
        metadata_states[str(model["metadata_state"])] += 1
        identity_kinds[str(model.get("identity_kind", ""))] += 1
        if model.get("counts_in_source_catalogue"):
            source_catalogue_total += 1
    return {
        "total": len(models),
        # ``total`` counts every registered identity, so it does not move when a
        # descriptor breaks. The health census is what moves, and the offending
        # models are named rather than left to be inferred from a count.
        "corpus_revision": corpus_revision(models),
        "metadata_states": metadata_states,
        # `total` is every registered identity. It is not the number of models
        # from the literature, and a browser that shows only the total invites
        # exactly that reading: the corpus mixes published models, project
        # originals, SC-compatibility identities and one API alias.
        "identity_kinds": dict(sorted(identity_kinds.items())),
        "source_catalogue_total": source_catalogue_total,
        "invalid_models": sorted(
            str(model["name"])
            for model in models
            if model["metadata_state"] == METADATA_STATE_INVALID
        ),
        "families": families,
        "maturities": dict(sorted(maturity_counts.items())),
        "behaviors": behaviors,
        "science_tiers": dict(sorted(science_tier_counts.items())),
        "silicon_tiers": dict(sorted(silicon_tier_counts.items())),
        "verified_science_tiers": dict(sorted(verified_science_counts.items())),
        "verified_silicon_tiers": dict(sorted(verified_silicon_counts.items())),
    }


#: Where the per-model reference pages are looked for, in order. The packaged
#: location is tried first so an installed distribution serves its own copy; the
#: checkout's `docs/api/models` is the fallback a working tree resolves to.
#: Neither is assumed to exist — :func:`documentation_root` says which, if
#: either, is actually there.
_PACKAGED_DOCS_DIR = Path(__file__).resolve().parent / "model_docs"
_CHECKOUT_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "api" / "models"
_DOCS_DIR = _CHECKOUT_DOCS_DIR


class ModelDocumentationUnavailable(RuntimeError):
    """Raised when no reference-page directory is installed at all.

    Distinct from a model simply having no page: the first is a packaging
    state that applies to every model, the second is a fact about one. Reporting
    the first as the second blames the model for the distribution.
    """


def documentation_root() -> Path | None:
    """Return the directory holding the per-model reference pages, or ``None``.

    Returns
    -------
    pathlib.Path or None
        The packaged directory when the distribution carries one, otherwise the
        checkout's ``docs/api/models`` when running from a working tree, and
        ``None`` when neither is present.
    """
    for candidate in (_PACKAGED_DOCS_DIR, _CHECKOUT_DOCS_DIR):
        if candidate.is_dir():
            return candidate
    return None


def model_documentation(name: str) -> dict[str, Any] | None:
    """Return the rendered reference documentation for a model, or ``None``.

    The per-model reference page lives at ``docs/api/models/<module>.md`` in a
    checkout and at ``sc_neurocore/studio/model_docs/<module>.md`` in a
    distribution that packages them. The Studio serves the Markdown so the
    documentation is browsable next to the live model.

    Returns
    -------
    dict or None
        The page, or ``None`` when this model has none.

    Raises
    ------
    ModelDocumentationUnavailable
        When no reference-page directory is installed. Every model is then
        undocumented for the same reason, which is a fact about the
        distribution and not about any model.
    """
    if name not in _CLASS_TO_MODULE:
        return None
    root = documentation_root()
    if root is None:
        raise ModelDocumentationUnavailable(
            "no model reference pages are installed: this distribution packages "
            "none and no checkout was found beside it"
        )
    module = _CLASS_TO_MODULE[name]
    path = root / f"{module}.md"
    if not path.is_file():
        return None
    return {"name": name, "slug": f"models/{module}", "markdown": path.read_text(encoding="utf-8")}
