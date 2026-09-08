# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from sc_neurocore.neurons.model_identity import identity_registry
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.model_catalogue import list_models, model_facets


def test_every_entry_states_which_kind_of_identity_it_is() -> None:
    """A browser showing only a total invites every model to be read as published.

    The corpus mixes published literature, project originals and
    SC-compatibility identities. Until this, the Studio said 185 and nothing
    else, while the identity ledger had recorded the split all along.
    """
    kinds = {str(row["identity_kind"]) for row in list_models()}

    assert kinds <= {"source-literature", "project-original", "sc-compatibility", "api-alias"}
    assert "" not in kinds


def test_the_census_matches_the_identity_registry() -> None:
    """The catalogue's split is the registry's, not a second count beside it."""
    facets = model_facets()
    registry = identity_registry()
    expected: dict[str, int] = {}
    for name in _CLASS_TO_MODULE:
        identity = registry.get(name)
        if identity is not None:
            expected[identity.kind] = expected.get(identity.kind, 0) + 1

    assert facets["identity_kinds"] == dict(sorted(expected.items()))


def test_the_source_catalogue_total_is_smaller_than_the_registered_total() -> None:
    """`total` counts registered identities; it is not the catalogue figure."""
    facets = model_facets()

    assert facets["source_catalogue_total"] < facets["total"]
    assert facets["source_catalogue_total"] == sum(
        1 for row in list_models() if row["counts_in_source_catalogue"]
    )


def test_an_api_alias_does_not_inflate_the_literature_count() -> None:
    """The backlog's own failure case, asserted rather than assumed.

    `KilincBhattMapNeuron` is an alias of `SCAdaptiveThresholdMapNeuron`. It is
    an identity in the registry and not a registered catalogue model, so it
    reaches no catalogue row and cannot be counted as a published model.
    """
    registry = identity_registry()
    aliases = [name for name, identity in registry.items() if identity.kind == "api-alias"]

    assert aliases, "the fixture assumes the corpus still holds an alias"
    for alias in aliases:
        assert alias not in _CLASS_TO_MODULE
        assert registry[alias].canonical_class != alias

    listed = {str(row["name"]) for row in list_models()}
    assert listed.isdisjoint(aliases)


def test_an_entry_the_registry_does_not_hold_says_so_rather_than_guessing() -> None:
    """An unclassified row reports empty fields, never a plausible default."""
    from sc_neurocore.studio.model_catalogue import _identity_fields

    assert _identity_fields("NoSuchNeuron") == {
        "identity_kind": "",
        "counts_in_source_catalogue": False,
        "public_label": "",
        "aliases": [],
    }
