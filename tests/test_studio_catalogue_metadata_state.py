# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.studio import model_catalogue as catalogue

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def broken_descriptor() -> Iterator[str]:
    """Make one model's descriptor load raise, the way a corrupt file would."""
    catalogue._models_cache = None
    victim = sorted(catalogue._CLASS_TO_MODULE)[0]
    real = catalogue.load_descriptor

    def _load(name: str) -> Any:
        if name == victim:
            raise ValueError("descriptor is not valid TOML")
        return real(name)

    catalogue.load_descriptor = _load  # type: ignore[assignment]
    catalogue._models_cache = None
    try:
        yield victim
    finally:
        catalogue.load_descriptor = real  # type: ignore[assignment]
        catalogue._models_cache = None


def _entry(name: str) -> dict[str, Any]:
    return next(row for row in catalogue.list_models() if row["name"] == name)


def test_every_registered_model_is_listed() -> None:
    """The catalogue enumerates the registry, so a count cannot hide an omission."""
    catalogue._models_cache = None
    names = {str(row["name"]) for row in catalogue.list_models()}

    assert names == set(catalogue._CLASS_TO_MODULE)


def test_a_healthy_entry_declares_available_metadata() -> None:
    """A committed descriptor is reported as declared metadata, with no error."""
    catalogue._models_cache = None
    row = catalogue.list_models()[0]

    assert row["metadata_state"] == catalogue.METADATA_STATE_AVAILABLE
    assert row["metadata_error"] is None


def test_unreadable_metadata_is_reported_not_omitted(broken_descriptor: str) -> None:
    """The acceptance case: a diagnostic, never a smaller silent success count."""
    rows = catalogue.list_models()
    row = _entry(broken_descriptor)

    assert len(rows) == len(catalogue._CLASS_TO_MODULE)
    assert row["metadata_state"] == catalogue.METADATA_STATE_INVALID
    assert row["metadata_error"] == "ValueError: descriptor is not valid TOML"


def test_an_invalid_entry_carries_every_key_a_healthy_entry_carries(
    broken_descriptor: str,
) -> None:
    """Key parity keeps every consumer working on a corpus with a fault in it."""
    rows = catalogue.list_models()
    broken = _entry(broken_descriptor)
    healthy = next(
        row for row in rows if row["metadata_state"] == catalogue.METADATA_STATE_AVAILABLE
    )

    assert set(broken) == set(healthy)


def test_facets_census_names_the_invalid_models(broken_descriptor: str) -> None:
    """Corpus health is a census plus names, not a total that quietly shrinks."""
    facets = catalogue.model_facets()

    assert facets["total"] == len(catalogue._CLASS_TO_MODULE)
    assert facets["metadata_states"][catalogue.METADATA_STATE_INVALID] == 1
    assert facets["invalid_models"] == [broken_descriptor]


def test_corpus_revision_separates_a_healthy_corpus_from_a_degraded_one() -> None:
    """A revision that ignored health would call two different corpora the same."""
    catalogue._models_cache = None
    healthy = catalogue.corpus_revision(catalogue.list_models())
    degraded = catalogue.corpus_revision(
        [
            {**row, "metadata_state": catalogue.METADATA_STATE_INVALID} if index == 0 else row
            for index, row in enumerate(catalogue.list_models())
        ]
    )

    assert healthy != degraded
    assert len(healthy) == 16


def test_corpus_revision_is_stable_across_entry_order() -> None:
    """Clients comparing revisions must not disagree over listing order."""
    catalogue._models_cache = None
    models = catalogue.list_models()

    assert catalogue.corpus_revision(models) == catalogue.corpus_revision(models[::-1])


def test_conformance_scope_keeps_a_model_whose_metadata_broke(
    broken_descriptor: str,
) -> None:
    """A list-scoped gate must not go quiet about a model that left the list.

    ``tools/runtime_state_conformance.py`` derives its matrix scope from
    :func:`list_models`. Before this contract a metadata fault removed the model
    from the catalogue and from the gate's scope together, so the gate stopped
    checking exactly the model that had just broken.
    """
    spec = importlib.util.spec_from_file_location(
        "runtime_state_conformance", REPO_ROOT / "tools" / "runtime_state_conformance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    names = module.model_names()

    assert len(names) == len(catalogue._CLASS_TO_MODULE)
    assert broken_descriptor in names
