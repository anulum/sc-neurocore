# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio catalogue query: indexed filtering and drill-down facets

"""Filter the model catalogue on the server, with facet counts that drill down.

The catalogue grows, so the browser does not hold the only copy of the
filtering rule: a query names the text, family, behaviour, identity kind and
metadata state it wants and the readiness it requires, and the answer is the
matching identities with the facet counts a reader chooses between next.

Readiness filters read the **verified** tiers only — those bound to facet
receipts whose subjects still match the repository — never the tiers a
descriptor declares, so "at least S3" means proven at S3.

Facet counts are disjunctive: each facet is counted over the models every
*other* filter admits, so choosing one family still shows how many models each
other family would give, instead of zeroing them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sc_neurocore.studio.model_catalogue import corpus_revision, list_models

CATALOGUE_QUERY_SCHEMA_VERSION = "sc-neurocore.studio.catalogue-query.v1"
_SCIENCE_TIERS = 5
_SILICON_TIERS = 5
_FACETS = ("family", "behavior", "identity_kind", "metadata_state")


class CatalogueQueryRejected(ValueError):
    """A query names a filter value the catalogue cannot hold."""


@dataclass(frozen=True, slots=True)
class CatalogueQuery:
    """What a catalogue query asks for; every field left at its default is no filter."""

    text: str = ""
    family: str = ""
    behavior: str = ""
    identity_kind: str = ""
    metadata_state: str = ""
    min_verified_science: int = 0
    min_verified_silicon: int = 0
    verified_perfect_only: bool = False

    @classmethod
    def from_params(cls, params: Mapping[str, str]) -> CatalogueQuery:
        """Build a query from HTTP query parameters, refusing what cannot be one.

        Raises
        ------
        CatalogueQueryRejected
            When a parameter is unknown, a tier is not an integer in range, or
            ``verified_perfect_only`` is not ``true`` or ``false``.
        """
        unknown = sorted(set(params) - set(cls.__dataclass_fields__))
        if unknown:
            raise CatalogueQueryRejected(f"unknown catalogue query parameter: {', '.join(unknown)}")
        perfect = params.get("verified_perfect_only", "false")
        if perfect not in ("true", "false"):
            raise CatalogueQueryRejected("verified_perfect_only must be true or false")
        return cls(
            text=params.get("text", ""),
            family=params.get("family", ""),
            behavior=params.get("behavior", ""),
            identity_kind=params.get("identity_kind", ""),
            metadata_state=params.get("metadata_state", ""),
            min_verified_science=_tier(params, "min_verified_science", _SCIENCE_TIERS),
            min_verified_silicon=_tier(params, "min_verified_silicon", _SILICON_TIERS),
            verified_perfect_only=perfect == "true",
        )


def _tier(params: Mapping[str, str], name: str, highest: int) -> int:
    raw = params.get(name, "0")
    if not raw.isdigit() or int(raw) > highest:
        raise CatalogueQueryRejected(f"{name} must be an integer from 0 to {highest}")
    return int(raw)


@dataclass(frozen=True, slots=True)
class _CatalogueIndex:
    """Name sets per facet value, built once per catalogue listing."""

    names: frozenset[str]
    postings: Mapping[str, Mapping[str, frozenset[str]]]
    search_text: Mapping[str, str]
    verified_science: Mapping[str, int]
    verified_silicon: Mapping[str, int | None]
    verified_perfect: frozenset[str]


_index_cache: tuple[int, _CatalogueIndex] | None = None


def _facet_values(model: Mapping[str, Any], facet: str) -> Iterable[str]:
    if facet == "behavior":
        return [str(tag) for tag in model.get("behavior_tags", [])]
    return [str(model.get(facet, ""))]


def _catalogue_index(models: list[dict[str, Any]]) -> _CatalogueIndex:
    """Return the index of ``models``, rebuilt only when the listing is a new one."""
    global _index_cache
    if _index_cache is not None and _index_cache[0] == id(models):
        return _index_cache[1]
    postings: dict[str, dict[str, set[str]]] = {facet: {} for facet in _FACETS}
    for model in models:
        for facet in _FACETS:
            for value in _facet_values(model, facet):
                postings[facet].setdefault(value, set()).add(str(model["name"]))
    index = _CatalogueIndex(
        names=frozenset(str(model["name"]) for model in models),
        postings={
            facet: {value: frozenset(names) for value, names in values.items()}
            for facet, values in postings.items()
        },
        search_text={
            str(model["name"]): " ".join(
                [
                    str(model["name"]),
                    str(model.get("public_label", "")),
                    " ".join(str(alias) for alias in model.get("aliases", [])),
                    str(model.get("family", "")),
                    str(model.get("category", "")),
                    str(model.get("description", "")),
                ]
            ).lower()
            for model in models
        },
        verified_science={
            str(model["name"]): int(model.get("verified_science_tier", 0)) for model in models
        },
        verified_silicon={
            str(model["name"]): model.get("verified_silicon_tier") for model in models
        },
        verified_perfect=frozenset(
            str(model["name"]) for model in models if model.get("is_perfect_verified") is True
        ),
    )
    _index_cache = (id(models), index)
    return index


def _admitted(index: _CatalogueIndex, query: CatalogueQuery, *, skip: str = "") -> frozenset[str]:
    """Return the names every filter admits, except the facet named by ``skip``."""
    names = index.names
    for facet in _FACETS:
        value = getattr(query, facet)
        if value and facet != skip:
            names = names & index.postings[facet].get(value, frozenset())
    if query.text:
        needle = query.text.lower()
        names = frozenset(name for name in names if needle in index.search_text[name])
    if query.min_verified_science:
        names = frozenset(
            name for name in names if index.verified_science[name] >= query.min_verified_science
        )
    if query.min_verified_silicon:
        names = frozenset(
            name for name in names if _silicon_at_least(index, name, query.min_verified_silicon)
        )
    if query.verified_perfect_only:
        names = names & index.verified_perfect
    return names


def _silicon_at_least(index: _CatalogueIndex, name: str, floor: int) -> bool:
    tier = index.verified_silicon[name]
    # A model not enrolled on the silicon axis has no tier and meets no floor.
    return tier is not None and tier >= floor


def _counts(index: _CatalogueIndex, names: frozenset[str], facet: str) -> dict[str, int]:
    return {
        value: len(members & names)
        for value, members in sorted(index.postings[facet].items())
        if members & names
    }


def _tier_counts(names: frozenset[str], tier_of: Callable[[str], int | None]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for name in names:
        tier = tier_of(name)
        counts["none" if tier is None else str(tier)] += 1
    return dict(sorted(counts.items()))


def query_catalogue(query: CatalogueQuery) -> dict[str, Any]:
    """Return the identities ``query`` admits and the facet counts around them.

    Returns
    -------
    dict
        ``schema_version``, the ``corpus_revision`` the answer was computed on,
        ``total`` registered identities, ``matched`` count, the matching
        ``models`` (names, sorted) and ``facets``: for each facet, the count per
        value over the models every other filter admits; for the verified tiers
        and ``verified_perfect``, counts over the matched models.
    """
    models = list_models()
    index = _catalogue_index(models)
    matched = _admitted(index, query)
    facets: dict[str, Any] = {
        facet: _counts(index, _admitted(index, query, skip=facet), facet) for facet in _FACETS
    }
    facets["verified_science_tiers"] = _tier_counts(matched, index.verified_science.__getitem__)
    facets["verified_silicon_tiers"] = _tier_counts(matched, index.verified_silicon.__getitem__)
    facets["verified_perfect"] = len(matched & index.verified_perfect)
    return {
        "schema_version": CATALOGUE_QUERY_SCHEMA_VERSION,
        "corpus_revision": corpus_revision(models),
        "total": len(index.names),
        "matched": len(matched),
        "models": sorted(matched),
        "facets": facets,
    }


__all__ = [
    "CATALOGUE_QUERY_SCHEMA_VERSION",
    "CatalogueQuery",
    "CatalogueQueryRejected",
    "query_catalogue",
]
