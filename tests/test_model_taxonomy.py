# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Model family taxonomy gate

"""Gate for the curated neuron model family taxonomy.

Every registered model must be classified into exactly one family with a stable
category slug, so the discovery taxonomy cannot fall behind the registry as the
library grows.
"""

from __future__ import annotations

import re
from collections import Counter

from sc_neurocore.neurons.model_taxonomy import (
    _FAMILIES,
    classified_models,
    families,
    model_family,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def test_every_registered_model_is_classified() -> None:
    """Tier 1: no registered model is missing from the family taxonomy."""

    unclassified = sorted(set(_CLASS_TO_MODULE) - classified_models())
    assert unclassified == [], f"models with no family classification: {unclassified}"


def test_taxonomy_has_no_extras_or_duplicates() -> None:
    """The taxonomy classifies only real models, each exactly once."""

    members = [name for _family, (_slug, names) in _FAMILIES.items() for name in names]
    duplicates = sorted(name for name, count in Counter(members).items() if count > 1)
    assert duplicates == [], f"models classified into more than one family: {duplicates}"
    extras = sorted(classified_models() - set(_CLASS_TO_MODULE))
    assert extras == [], f"taxonomy classifies unregistered models: {extras}"


def test_family_categories_are_unique_slugs() -> None:
    """Each family declares a unique, well-formed category slug."""

    slugs = list(families().values())
    assert all(_SLUG.fullmatch(slug) for slug in slugs), slugs
    assert len(slugs) == len(set(slugs)), "duplicate category slugs across families"


def test_model_family_returns_family_and_slug() -> None:
    """A classified model returns its family display name and category slug."""

    family, category = model_family("GolgiCell")  # type: ignore[misc]
    assert family == "Cerebellar"
    assert category == "cerebellar"
    assert model_family("DefinitelyNotARegisteredModel") is None
