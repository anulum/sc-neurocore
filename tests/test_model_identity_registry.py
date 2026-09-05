# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Identity registry contract: every class resolves unambiguously

"""Contract tests for the canonical model identity registry."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from sc_neurocore.neurons import model_identity
from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.model_identity import (
    NETWORK_IDENTITIES,
    ModelIdentityError,
    catalogue_counts,
    identity_registry,
    iter_source_catalogue,
    public_fidelity_bindings,
    resolve_identity,
    schema_for_class,
)
from sc_neurocore.neurons.model_taxonomy import _COMPATIBILITY_ALIASES, classified_models
from sc_neurocore.neurons.models import _CLASS_TO_MODULE

_SCHEMA_DIR = Path(model_identity.__file__).resolve().parent / "model_schemas"


def test_every_registered_class_and_alias_has_exactly_one_identity() -> None:
    """Tier 1: registry keys are the registered classes plus the import aliases."""
    registry = identity_registry()
    expected = set(_CLASS_TO_MODULE) | set(_COMPATIBILITY_ALIASES)
    assert set(registry) == expected
    assert all(record.class_name == name for name, record in registry.items())


def test_identity_kinds_follow_the_declared_rules() -> None:
    """SC classes are compatibility identities; aliases resolve; others count."""
    registry = identity_registry()
    for name, record in registry.items():
        if name in _COMPATIBILITY_ALIASES:
            assert record.kind == "api-alias"
            assert record.canonical_class == _COMPATIBILITY_ALIASES[name]
            assert not record.counts_in_source_catalogue
        elif name.startswith("SC"):
            assert record.kind == "sc-compatibility"
            assert record.source.doi == ""
            assert not record.counts_in_source_catalogue
        else:
            assert record.kind in {"source-literature", "project-original"}
            assert record.counts_in_source_catalogue
            assert record.canonical_class == name
        if record.kind == "source-literature":
            assert record.source.basis in {"doi", "url", "citation-unlocated"}
        if record.kind == "project-original":
            assert record.source.basis == "project-specification"


def test_source_catalogue_excludes_aliases_and_sc_identities() -> None:
    """The source count is the registry minus SC identities; aliases never count."""
    counts = catalogue_counts()
    sc_classes = [name for name in _CLASS_TO_MODULE if name.startswith("SC")]
    assert counts.registered == len(_CLASS_TO_MODULE)
    assert counts.sc_compatibility == len(sc_classes)
    assert counts.api_aliases == len(_COMPATIBILITY_ALIASES)
    assert counts.source_catalogue == len(_CLASS_TO_MODULE) - len(sc_classes)
    assert counts.source_catalogue == counts.source_literature + counts.project_original
    assert counts.remaining_source == counts.source_catalogue - counts.polyglot_complete_source
    assert {record.class_name for record in iter_source_catalogue()} == {
        name for name in _CLASS_TO_MODULE if not name.startswith("SC")
    }


def test_every_schema_stem_binds_to_exactly_one_class() -> None:
    """No schema profile is silently unowned or claimed twice."""
    registry = identity_registry()
    stems = {path.stem for path in _SCHEMA_DIR.iterdir() if path.suffix in {".toml", ".json"}}
    owners: dict[str, str] = {}
    for record in registry.values():
        for profile in record.schema_profiles:
            assert profile.stem not in owners, f"{profile.stem} bound twice"
            owners[profile.stem] = record.class_name
    assert set(owners) == stems
    assert catalogue_counts().schema_profiles == len(stems)


def test_alias_resolves_to_its_canonical_identity() -> None:
    """Historical import names resolve without becoming catalogue rows."""
    for alias, canonical in _COMPATIBILITY_ALIASES.items():
        assert resolve_identity(alias).class_name == canonical
        assert alias in identity_registry()[canonical].aliases
    with pytest.raises(ModelIdentityError):
        resolve_identity("NoSuchNeuronAnywhere")


def test_taxonomy_classifies_every_counted_and_sc_identity() -> None:
    """Every registered class carries a family; aliases inherit their target's."""
    registry = identity_registry()
    for name in _CLASS_TO_MODULE:
        assert registry[name].family, f"{name} has no taxonomy family"
    assert classified_models() == set(_CLASS_TO_MODULE)
    for alias, canonical in _COMPATIBILITY_ALIASES.items():
        assert registry[alias].family == registry[canonical].family


def test_public_bindings_name_registered_classes_and_unique_labels() -> None:
    """Public fidelity rows bind to registered classes with unique labels."""
    bindings = public_fidelity_bindings()
    labels = [label for label, _status in bindings.values()]
    assert len(labels) == len(set(labels))
    assert set(bindings) <= set(_CLASS_TO_MODULE)
    registry = identity_registry()
    for name, (label, status) in bindings.items():
        assert registry[name].public_label == label
        assert registry[name].public_status == status
    for name, record in registry.items():
        if name in bindings:
            continue
        if record.kind == "api-alias":
            assert record.public_status == "alias"
        elif record.kind == "sc-compatibility":
            assert record.public_status == "unlisted"
        else:
            assert record.public_status == "remaining"


def test_strict_promotion_carries_an_explicit_revalidation_status() -> None:
    """Every promoted identity is receipt-bound or explicitly not revalidated."""
    registry = identity_registry()
    counts = catalogue_counts()
    promoted = [r for r in registry.values() if r.public_status == "polyglot-complete"]
    assert promoted, "no promoted identities bound"
    for record in promoted:
        assert record.revalidation in {"receipt-bound", "not-revalidated"}
    promoted_source = [r for r in promoted if r.counts_in_source_catalogue]
    assert counts.polyglot_complete_source == len(promoted_source)
    assert counts.polyglot_complete_sc == len(promoted) - len(promoted_source)
    assert counts.receipt_bound_complete + counts.not_revalidated_complete == len(promoted_source)
    for record in registry.values():
        if record.public_status != "polyglot-complete":
            assert record.revalidation == "not-completed"


def test_missing_gates_are_explicit_rows_not_omissions() -> None:
    """Open evidence gates are named per identity; nothing is dropped silently."""
    registry = identity_registry()
    known = {
        "source-locator",
        "schema-dsl-profile",
        "descriptor",
        "independent-source-validation",
        "rtl-compile",
        "cosim",
        "synthesis",
        "timing",
        "formal-equivalence",
        "ppa",
    } | {f"backend:{name}" for name in model_identity.REQUIRED_BACKENDS}
    for record in registry.values():
        assert set(record.missing_gates) <= known, record.class_name
        assert len(record.missing_gates) == len(set(record.missing_gates))
        if record.kind == "api-alias":
            assert record.missing_gates == ()
        else:
            assert "descriptor" not in record.missing_gates, (
                f"{record.class_name} lacks a descriptor"
            )


def test_missing_gates_follow_the_descriptor_claims() -> None:
    """Claimed facets clear their gate; unclaimed facets and absent descriptors do not."""
    source = model_identity.SourceLocator(basis="doi", doi="10.1000/x")
    profile = (model_identity.SchemaProfile(stem="x", basis="alias-table"),)
    complete = {
        "backends": {name: {"status": "implemented"} for name in model_identity.REQUIRED_BACKENDS},
        "validation": {"dynamics_faithful": True},
        "silicon": {
            "compiles": True,
            "cosim_validated": True,
            "synthesised": True,
            "timing_closed": True,
            "formally_equivalent": True,
            "ppa_signed": True,
        },
    }
    assert model_identity._missing_gates(complete, profile, source, "source-literature") == ()
    assert model_identity._missing_gates(None, (), source, "source-literature") == (
        "schema-dsl-profile",
        "descriptor",
    )
    unlocated = model_identity.SourceLocator(basis="citation-unlocated", authors=("Someone",))
    gates = model_identity._missing_gates({}, profile, unlocated, "source-literature")
    assert gates[0] == "source-locator"
    assert set(gates) >= {"backend:mojo", "independent-source-validation", "ppa"}


def test_network_identities_import_and_name_a_registered_cell() -> None:
    """Network-level identities exist in code and point at a registered cell."""
    assert NETWORK_IDENTITIES
    for network in NETWORK_IDENTITIES:
        module = importlib.import_module(network.module)
        assert hasattr(module, network.class_name)
        assert network.cell_identity in _CLASS_TO_MODULE
        assert network.kind == "sc-compatibility" or not network.class_name.startswith("SC")
    assert catalogue_counts().network_identities == len(NETWORK_IDENTITIES)


def test_sc_identity_with_a_literature_doi_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A count-neutral SC identity carrying a DOI is ambiguous and fails closed."""
    identity_registry.cache_clear()
    original = load_descriptor_payload

    def poisoned(class_name: str) -> dict[str, object] | None:
        payload = original(class_name)
        if class_name == "SCLapicqueLIFNeuron" and payload is not None:
            payload = dict(payload)
            payload["provenance"] = {"doi": "10.1000/fake"}
        return payload

    monkeypatch.setattr(model_identity, "load_descriptor_payload", poisoned)
    try:
        with pytest.raises(ModelIdentityError, match="SCLapicqueLIFNeuron"):
            identity_registry()
    finally:
        identity_registry.cache_clear()


def test_public_binding_to_an_unregistered_class_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page row bound to an unknown class is an error, not a dropped row."""
    identity_registry.cache_clear()
    bindings = dict(model_identity._PUBLIC_FIDELITY_ROWS)
    bindings["GhostNeuron"] = ("Ghost", "polyglot-complete")
    monkeypatch.setattr(model_identity, "_PUBLIC_FIDELITY_ROWS", bindings)
    try:
        with pytest.raises(ModelIdentityError, match="GhostNeuron"):
            identity_registry()
    finally:
        identity_registry.cache_clear()


def test_unbound_schema_stem_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema stem that joins no registered class fails instead of vanishing."""
    identity_registry.cache_clear()
    original = model_identity._schema_stems

    def with_orphan() -> tuple[str, ...]:
        return (*original(), "orphan_profile_stem")

    monkeypatch.setattr(model_identity, "_schema_stems", with_orphan)
    try:
        with pytest.raises(ModelIdentityError, match="orphan_profile_stem"):
            identity_registry()
    finally:
        identity_registry.cache_clear()


def test_schema_for_class_separates_identities_sharing_a_module() -> None:
    """A retained SC identity never inherits the source profile of its module."""
    assert schema_for_class("LapicqueNeuron") == "lapicque"
    assert schema_for_class("SCLapicqueLIFNeuron") == "sc_lapicque_lif"
    assert schema_for_class("PerfectIntegratorNeuron") == "perfect_integrator"
    assert schema_for_class("SCInclusivePerfectIntegratorNeuron") == "sc_perfect_integrator"
    assert schema_for_class("KilincBhattMapNeuron") == schema_for_class(
        "SCAdaptiveThresholdMapNeuron"
    )
    with pytest.raises(ModelIdentityError):
        schema_for_class("NoSuchNeuronAnywhere")
