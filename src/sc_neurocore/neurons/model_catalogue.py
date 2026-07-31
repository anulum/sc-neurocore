# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — On-disk model descriptor corpus access and coverage

"""Load committed model descriptors and report curation coverage.

The descriptor corpus lives at ``neurons/model_descriptors/<ClassName>.toml`` —
one declarative descriptor per registered model, the durable curation surface.
This module reads and validates them, and reports how complete the catalogue's
metadata is (per completeness tier) so coverage can be tracked as the library is
tuned. The structural fields of each committed descriptor are kept in sync with
the model code by the corpus tool and the completeness gate, so the corpus never
drifts from the implementation.
"""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.neurons.descriptor_tiers import SILICON_RUNGS, completeness_tiers
from sc_neurocore.neurons.model_descriptor import (
    ModelDescriptor,
    descriptor_completeness_tier,
    parse_model_descriptor,
)
from sc_neurocore.neurons.model_taxonomy import canonical_model_name
from sc_neurocore.neurons.models import _CLASS_TO_MODULE

DESCRIPTOR_DIR = Path(__file__).resolve().parent / "model_descriptors"


def descriptor_path(class_name: str) -> Path:
    """Return the committed descriptor path for a model class.

    Parameters
    ----------
    class_name:
        Public Python class identifier for the model descriptor.

    Returns
    -------
    pathlib.Path
        Absolute path under ``neurons/model_descriptors``.

    Raises
    ------
    ValueError
        If ``class_name`` is empty, private, dotted, or path-like.
    """
    if not class_name.isidentifier() or class_name.startswith("_"):
        raise ValueError("model class name must be a public Python identifier")
    return DESCRIPTOR_DIR / f"{canonical_model_name(class_name)}.toml"


def load_descriptor_payload(class_name: str) -> dict[str, Any] | None:
    """Return the raw committed descriptor payload.

    Parameters
    ----------
    class_name:
        Public Python class identifier for the model descriptor.

    Returns
    -------
    dict[str, Any] | None
        Parsed TOML payload when the descriptor is committed, otherwise
        ``None``.

    Raises
    ------
    ValueError
        If ``class_name`` is empty, private, dotted, or path-like.
    """
    path = descriptor_path(class_name)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload: dict[str, Any] = tomllib.load(handle)
    return payload


def load_descriptor(class_name: str) -> ModelDescriptor | None:
    """Return the validated committed descriptor for a model.

    Parameters
    ----------
    class_name:
        Public Python class identifier for the model descriptor.

    Returns
    -------
    ModelDescriptor | None
        Validated descriptor when the TOML file exists, otherwise ``None``.

    Raises
    ------
    ValueError
        If ``class_name`` is empty, private, dotted, or path-like.
    """
    payload = load_descriptor_payload(class_name)
    if payload is None:
        return None
    return parse_model_descriptor(payload)


@dataclass(frozen=True, slots=True)
class CatalogueCoverage:
    """Aggregate descriptor coverage across the registered model catalogue.

    Parameters
    ----------
    total_models:
        Number of registered models.
    described:
        Number of models with a committed descriptor.
    tier_counts:
        Count of described models at each science-kernel tier (keys ``0``-``3``);
        the legacy curation view, retained for backward compatibility.
    science_tier_counts:
        Count of described models at each full science-axis tier (keys ``0``-``5``,
        S0-S5).
    silicon_tier_counts:
        Count of described models at each silicon-axis tier, keyed by label
        (``"none"`` for no compile-clean RTL, then ``"H0"``-``"H5"``).
    citeable:
        Number of described models with citeable provenance.
    fully_curated_parameters:
        Number of described models whose every parameter has unit, range, and
        meaning.
    """

    total_models: int
    described: int
    tier_counts: dict[int, int]
    science_tier_counts: dict[int, int]
    silicon_tier_counts: dict[str, int]
    citeable: int
    fully_curated_parameters: int

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-compatible coverage summary.

        Returns
        -------
        dict[str, object]
            Stable public summary with sorted tier-count keys, both readiness
            axes, and an ``undescribed`` derived count.
        """
        return {
            "total_models": self.total_models,
            "described": self.described,
            "undescribed": self.total_models - self.described,
            "tier_counts": dict(sorted(self.tier_counts.items())),
            "science_tier_counts": dict(sorted(self.science_tier_counts.items())),
            "silicon_tier_counts": dict(self.silicon_tier_counts),
            "citeable": self.citeable,
            "fully_curated_parameters": self.fully_curated_parameters,
        }


def catalogue_descriptor_coverage() -> CatalogueCoverage:
    """Return descriptor coverage over every registered model.

    Returns
    -------
    CatalogueCoverage
        Aggregate coverage over the current model registry and committed
        descriptor corpus.
    """
    tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    science_counts = {tier: 0 for tier in range(6)}
    silicon_counts = {label: 0 for label in ("none", *SILICON_RUNGS)}
    described = 0
    citeable = 0
    fully_curated = 0
    for class_name in _CLASS_TO_MODULE:
        descriptor = load_descriptor(class_name)
        if descriptor is None:
            continue
        described += 1
        tier_counts[descriptor_completeness_tier(descriptor)] += 1
        tiers = completeness_tiers(descriptor)
        science_counts[tiers.science] += 1
        silicon_counts[tiers.silicon_label] += 1
        if descriptor.provenance.is_citeable:
            citeable += 1
        if descriptor.parameters and all(p.is_curated for p in descriptor.parameters):
            fully_curated += 1
    return CatalogueCoverage(
        total_models=len(_CLASS_TO_MODULE),
        described=described,
        tier_counts=tier_counts,
        science_tier_counts=science_counts,
        silicon_tier_counts=silicon_counts,
        citeable=citeable,
        fully_curated_parameters=fully_curated,
    )


__all__ = [
    "DESCRIPTOR_DIR",
    "CatalogueCoverage",
    "catalogue_descriptor_coverage",
    "descriptor_path",
    "load_descriptor",
    "load_descriptor_payload",
]
