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
from pathlib import Path
from typing import Any

from sc_neurocore.neurons.descriptor_tiers import completeness_tiers, is_perfect
from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import (
    ModelDescriptor,
    descriptor_completeness_tier,
)
from sc_neurocore.neurons.model_profile import resolve_profile
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.neurons.schema_module_aliases import schema_for_module
from sc_neurocore.neurons.universal_dsl import load_schema
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


class ModelMetadataError(RuntimeError):
    """Raised when Studio model metadata loading fails for a known model."""


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
    return {
        "name": descriptor.class_name,
        "module": descriptor.module,
        "metadata_state": METADATA_STATE_AVAILABLE,
        "metadata_error": None,
        "tier": tier,
        "evidence_kind": _evidence_kind(tier),
        "science_tier": tiers.science,
        "science_label": tiers.science_label,
        "silicon_tier": tiers.silicon,
        "silicon_label": tiers.silicon_label,
        **_verified_summary(descriptor.class_name),
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
    return {
        "schema_name": schema_name,
        "default_integrator": default_integrator,
        "integrators": list(dict.fromkeys(integrators)),
        "cosim_integrators": [
            integrator
            for integrator in dict.fromkeys(integrators)
            if integrator in profile.lowering.cosim_methods
        ],
        "default_q_format": "Q8.8",
        "q_formats": ["Q8.8", "Q16.16"],
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
    from sc_neurocore.neurons.readiness import verify_model

    if class_name not in identity_registry():
        return {
            "verified_science_tier": 0,
            "verified_science_label": "S0",
            "verified_silicon_tier": None,
            "verified_silicon_label": "none",
            "verified_profile": None,
        }
    record = verify_model(class_name, profile=_selected_profile(class_name))
    return {
        "verified_science_tier": record.verified_science,
        "verified_science_label": record.verified_science_label,
        "verified_silicon_tier": record.verified_silicon,
        "verified_silicon_label": record.verified_silicon_label,
        "verified_profile": record.profile,
    }


def _verified_detail(class_name: str) -> dict[str, Any]:
    """Return the per-facet verification block for a model detail.

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


def _readiness_detail(descriptor: ModelDescriptor) -> dict[str, Any]:
    """Build the auditable dual-axis readiness view for a declared descriptor.

    Surfaces the declared science (S0-S5) and silicon (H0-H5) tiers together
    with the raw evidence facets that justify them, and next to them the
    verified tiers and per-facet statuses derived from facet receipts, so a
    reviewer can see exactly why a model sits where it does, whether it meets
    its declared deployability class, and how much of the claim is bound to an
    executed, still-fresh receipt.
    """
    tiers = completeness_tiers(descriptor)
    return {
        "science_tier": tiers.science,
        "science_label": tiers.science_label,
        "silicon_tier": tiers.silicon,
        "silicon_label": tiers.silicon_label,
        "verified": _verified_detail(descriptor.class_name),
        "is_perfect": is_perfect(descriptor),
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


def _unreadable_summary(name: str, reason: str) -> dict[str, Any]:
    """Catalogue entry for a model whose metadata could not be read.

    Carries the same keys as a healthy entry so every consumer keeps working on
    a corpus with a fault in it, and states the fault rather than omitting the
    row.

    Parameters
    ----------
    name : str
        The registered model identity.
    reason : str
        The failure, as the exception described it. Path-free: descriptor
        loading reports the model, not the file it came from.

    Returns
    -------
    dict
        A catalogue entry with ``metadata_state`` ``"invalid"``.
    """
    return {
        "name": name,
        "module": _CLASS_TO_MODULE[name],
        "metadata_state": METADATA_STATE_INVALID,
        "metadata_error": reason,
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
        except (TypeError, AttributeError, ValueError) as exc:
            entry = _unreadable_summary(name, f"{type(exc).__name__}: {exc}")
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
    for model in models:
        metadata_states[str(model["metadata_state"])] += 1
    return {
        "total": len(models),
        # ``total`` counts every registered identity, so it does not move when a
        # descriptor breaks. The health census is what moves, and the offending
        # models are named rather than left to be inferred from a count.
        "corpus_revision": corpus_revision(models),
        "metadata_states": metadata_states,
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


_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "api" / "models"


def model_documentation(name: str) -> dict[str, Any] | None:
    """Return the rendered reference documentation for a model, or ``None``.

    The per-model reference page lives at ``docs/api/models/<module>.md``; the
    Studio serves its Markdown so the documentation is browsable inline next to
    the live model rather than only in the built docs site.
    """
    if name not in _CLASS_TO_MODULE:
        return None
    module = _CLASS_TO_MODULE[name]
    path = _DOCS_DIR / f"{module}.md"
    if not path.is_file():
        return None
    return {"name": name, "slug": f"models/{module}", "markdown": path.read_text(encoding="utf-8")}
