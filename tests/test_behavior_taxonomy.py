# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Behaviour tag vocabulary tests

"""Tests for the controlled behaviour-tag vocabulary."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.behavior_taxonomy import (
    BEHAVIOR_TAGS,
    STOCHASTIC_SAFE_TAGS,
    behavior_tag_definition,
    validate_behavior_tags,
)


def test_every_tag_has_a_definition() -> None:
    """Every vocabulary tag carries a non-empty observable definition."""

    for tag in BEHAVIOR_TAGS:
        assert behavior_tag_definition(tag).strip()


def test_stochastic_safe_tags_are_in_the_vocabulary() -> None:
    """The stochastic-safe tags are a subset of the full vocabulary."""

    assert STOCHASTIC_SAFE_TAGS <= BEHAVIOR_TAGS


def test_unknown_tag_definition_rejected() -> None:
    """Requesting an unknown tag's definition raises."""

    with pytest.raises(ValueError, match="unknown behaviour tag"):
        behavior_tag_definition("teleporting")


def test_validate_returns_sorted_unique_tuple() -> None:
    """Validation de-duplicates and sorts the tags."""

    assert validate_behavior_tags(["tonic", "excitable", "tonic"]) == ("excitable", "tonic")


def test_validate_accepts_empty() -> None:
    """An empty collection is valid (a model with no measured behaviour)."""

    assert validate_behavior_tags([]) == ()


def test_validate_rejects_unknown_tag() -> None:
    """A tag outside the vocabulary is rejected."""

    with pytest.raises(ValueError, match="unknown behaviour tag: 'wobbling'"):
        validate_behavior_tags(["excitable", "wobbling"])


def test_validate_rejects_a_bare_string() -> None:
    """A single string is not accepted as a tag collection."""

    with pytest.raises(ValueError, match="iterable of strings"):
        validate_behavior_tags("tonic")


def test_validate_rejects_non_iterable() -> None:
    """A non-iterable value is rejected."""

    with pytest.raises(ValueError, match="iterable of strings"):
        validate_behavior_tags(42)


def test_validate_rejects_non_string_member() -> None:
    """A non-string member is rejected."""

    with pytest.raises(ValueError, match="must be a string"):
        validate_behavior_tags(["tonic", 7])


def test_validate_rejects_excitable_and_quiescent_together() -> None:
    """A model cannot be both excitable and quiescent."""

    with pytest.raises(ValueError, match="both 'excitable' and 'quiescent'"):
        validate_behavior_tags(["excitable", "quiescent"])


def test_validate_rejects_quiescent_with_a_firing_tag() -> None:
    """A quiescent model cannot also carry a firing-pattern tag."""

    with pytest.raises(ValueError, match="quiescent.*cannot also carry"):
        validate_behavior_tags(["quiescent", "bursting"])


def test_validate_accepts_a_realistic_tag_set() -> None:
    """A plausible measured set passes through unchanged but sorted."""

    assert validate_behavior_tags(["excitable", "rate-coded", "tonic"]) == (
        "excitable",
        "rate-coded",
        "tonic",
    )
