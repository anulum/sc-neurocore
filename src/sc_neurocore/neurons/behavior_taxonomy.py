# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Controlled vocabulary for measured behaviour tags

"""Controlled vocabulary for the measured behaviour facet of a model.

Behaviour tags are a *measured* facet: each one is asserted only when a
behaviour probe observes it under a reproducible simulation, never declared by
hand. To keep the facet honest every tag has an exact, observable definition in
terms of the firing-pattern classifier output across a current sweep, and the
catalogue may carry only tags from this vocabulary. A descriptor that names a
tag outside the vocabulary, or a probe that derives one, is rejected.

The definitions are deliberately conservative — a tag describes what was
*observed* under the tested constant-current drive over the probe window, not a
claim about the model's full dynamical repertoire. Tags whose classification
flips between two identical runs (stochastic models) are not asserted; such a
model carries only the sign-robust excitability tags plus ``stochastic``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

# tag -> one-line observable definition (the predicate the probe applies).
_BEHAVIOR_TAG_DEFINITIONS: dict[str, str] = {
    "excitable": "fired at least one spike under some tested drive",
    "quiescent": "fired no spikes at any tested drive over the probe window",
    "tonic": "fired regularly (low ISI variability) under some sustained drive",
    "adapting": "showed spike-frequency adaptation (ISIs lengthening over time)",
    "bursting": "showed a burst-pause structure (bimodal ISI) under some drive",
    "irregular": "fired with moderate ISI variability under some drive",
    "chaotic": "fired with high ISI variability under some drive",
    "phasic": "fired only one or two spikes despite sustained drive at some level",
    "rate-coded": "firing rate rose monotonically with drive (an increasing f-I curve)",
    "stochastic": "spike train was not reproducible between two identical runs",
}

BEHAVIOR_TAGS = frozenset(_BEHAVIOR_TAG_DEFINITIONS)

# Tags that may be asserted for a stochastic model: only the sign-robust
# excitability verdict and the stochasticity flag itself survive run-to-run, so
# the fine pattern tags (tonic/adapting/...) are withheld for such models.
STOCHASTIC_SAFE_TAGS = frozenset({"excitable", "quiescent", "stochastic"})


def behavior_tag_definition(tag: str) -> str:
    """Return the observable definition of a behaviour tag.

    Raises
    ------
    ValueError
        If ``tag`` is not in the controlled vocabulary.
    """

    try:
        return _BEHAVIOR_TAG_DEFINITIONS[tag]
    except KeyError:
        raise ValueError(f"unknown behaviour tag: {tag!r}") from None


def validate_behavior_tags(tags: object) -> tuple[str, ...]:
    """Validate a behaviour-tag collection and return a sorted, unique tuple.

    A model is either ``excitable`` or ``quiescent`` but never both, and a
    ``quiescent`` model cannot also carry a firing-pattern tag — these
    contradictions signal a corrupted or hand-edited tag set and are rejected.

    Parameters
    ----------
    tags:
        An iterable of tag strings (for example the ``behavior_tags`` field of a
        descriptor or the output of the probe).

    Returns
    -------
    tuple[str, ...]
        The tags, de-duplicated and sorted, ready to store or compare.

    Raises
    ------
    ValueError
        If ``tags`` is not an iterable of strings, names a tag outside the
        vocabulary, or carries a contradictory combination.
    """

    if isinstance(tags, str) or not _is_iterable(tags):
        raise ValueError(
            f"behaviour tags must be an iterable of strings, got {type(tags).__name__}"
        )
    collected: set[str] = set()
    for tag in cast(Iterable[object], tags):
        if not isinstance(tag, str):
            raise ValueError(f"behaviour tag must be a string, got {type(tag).__name__}")
        if tag not in BEHAVIOR_TAGS:
            raise ValueError(f"unknown behaviour tag: {tag!r}")
        collected.add(tag)
    _reject_contradictions(collected)
    return tuple(sorted(collected))


def _reject_contradictions(tags: set[str]) -> None:
    """Reject mutually exclusive tag combinations."""

    if "excitable" in tags and "quiescent" in tags:
        raise ValueError("behaviour tags cannot be both 'excitable' and 'quiescent'")
    firing_tags = {"tonic", "adapting", "bursting", "irregular", "chaotic", "phasic", "rate-coded"}
    if "quiescent" in tags and tags & firing_tags:
        raise ValueError(
            "a 'quiescent' model cannot also carry a firing-pattern tag: "
            f"{sorted(tags & firing_tags)}"
        )


def _is_iterable(value: object) -> bool:
    """Return whether ``value`` can be iterated.

    Probing an arbitrary ``object`` for iterability is the point of the guard,
    so ``iter`` is deliberately called on a non-iterable static type.
    """

    try:
        iter(value)  # type: ignore[call-overload]  # runtime iterability probe over object
    except TypeError:
        return False
    return True


__all__ = [
    "BEHAVIOR_TAGS",
    "STOCHASTIC_SAFE_TAGS",
    "behavior_tag_definition",
    "validate_behavior_tags",
]
