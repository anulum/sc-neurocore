# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio model catalogue metadata

"""Descriptor-driven model catalogue for Studio discovery UX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sc_neurocore.neurons.descriptor_tiers import completeness_tiers, is_perfect
from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import (
    ModelDescriptor,
    descriptor_completeness_tier,
)
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
        }
    )
    return detail


def _compile_configuration(descriptor: ModelDescriptor) -> dict[str, Any] | None:
    """Return the canonical schema-backed Studio compile choices, if available."""
    from sc_neurocore.neurons.model_identity import ModelIdentityError, schema_for_class

    try:
        schema_name = schema_for_class(descriptor.class_name)
    except ModelIdentityError:
        schema_name = schema_for_module(descriptor.module.rsplit(".", 1)[-1])
    try:
        schema = load_schema(schema_name)
    except (FileNotFoundError, ValueError):
        return None

    integration = schema.get("integration", {})
    default_integrator = str(integration.get("method", "euler"))
    if default_integrator not in SUPPORTED_METHODS:
        return None
    extensions = schema.get("extensions", {})
    declared = extensions.get("integrator_options", [default_integrator])
    integrators = [
        str(value) for value in declared if isinstance(value, str) and value in SUPPORTED_METHODS
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
            if integrator in {"euler", "map"}
        ],
        "default_q_format": "Q8.8",
        "q_formats": ["Q8.8", "Q16.16"],
    }


def _verified_summary(class_name: str) -> dict[str, Any]:
    """Return the verified (receipt-bound) tiers for a browse entry.

    The declared tiers above come from the descriptor's own flags; these come
    only from facet receipts whose subjects still match the repository, so a
    browse entry always shows both what is claimed and what is proven.
    """
    from sc_neurocore.neurons.model_identity import identity_registry
    from sc_neurocore.neurons.readiness import verify_model

    if class_name not in identity_registry():
        return {
            "verified_science_tier": 0,
            "verified_science_label": "S0",
            "verified_silicon_tier": None,
            "verified_silicon_label": "none",
        }
    record = verify_model(class_name)
    return {
        "verified_science_tier": record.verified_science,
        "verified_science_label": record.verified_science_label,
        "verified_silicon_tier": record.verified_silicon,
        "verified_silicon_label": record.verified_silicon_label,
    }


def _verified_detail(class_name: str) -> dict[str, Any]:
    """Return the per-facet verification block for a model detail."""
    from sc_neurocore.neurons.readiness import verify_model

    record = verify_model(class_name)
    return {
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


def list_models() -> list[dict[str, Any]]:
    """Return declared metadata for every registered neuron model.

    Each entry is built from the model's committed descriptor (family, category,
    maturity, provenance, parameter and state counts). Models without a descriptor
    fall back to code introspection with an ``inferred`` category. Results are
    cached after the first call.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    result = []
    for name in sorted(_CLASS_TO_MODULE.keys()):
        try:
            descriptor = load_descriptor(name)
            if descriptor is not None:
                result.append(_descriptor_summary(descriptor))
            else:
                result.append(_introspected_summary(name))
        except (TypeError, AttributeError, ValueError):
            continue
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
    """Return the catalogue facet taxonomy and counts for discovery UX."""
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
    return {
        "total": len(models),
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
