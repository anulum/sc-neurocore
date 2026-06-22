# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Model descriptor contract and generator tests

"""Tests for the declarative model descriptor (schema v2) and its generator."""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

from sc_neurocore.neurons.descriptor_generator import (
    generate_descriptor,
    generate_descriptor_payload,
)
from sc_neurocore.neurons.model_descriptor import (
    MODEL_DESCRIPTOR_SCHEMA_VERSION,
    ModelDescriptorError,
    descriptor_completeness_tier,
    parse_model_descriptor,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE


def _minimal_payload() -> dict[str, object]:
    return {
        "metadata": {
            "schema_version": 2,
            "name": "AdEx",
            "class_name": "AdExNeuron",
            "module": "adex",
        },
        "state": {"v": {"init": -65.0}},
        "parameters": {"tau": {"default": 20.0}},
        "integration": {"dt": 0.1},
    }


def test_parse_minimal_descriptor() -> None:
    """A minimal valid payload parses into a Tier-0 descriptor."""

    descriptor = parse_model_descriptor(_minimal_payload())
    assert descriptor.class_name == "AdExNeuron"
    assert descriptor.schema_version == MODEL_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor.display_name == "AdEx"
    assert [s.name for s in descriptor.state] == ["v"]
    assert [p.name for p in descriptor.parameters] == ["tau"]
    assert descriptor_completeness_tier(descriptor) == 0


def test_parse_rejects_missing_class_name() -> None:
    payload = _minimal_payload()
    del payload["metadata"]["class_name"]  # type: ignore[index]
    with pytest.raises(ModelDescriptorError, match="class_name"):
        parse_model_descriptor(payload)


def test_parse_rejects_unknown_schema_version() -> None:
    payload = _minimal_payload()
    payload["metadata"]["schema_version"] = 1  # type: ignore[index]
    with pytest.raises(ModelDescriptorError, match="schema_version"):
        parse_model_descriptor(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda m: m["metadata"].__setitem__("biophysical_detail", "quantum"),
            "biophysical_detail",
        ),
        (lambda m: m["metadata"].__setitem__("maturity", "legendary"), "maturity"),
        (lambda m: m["metadata"].__setitem__("category", "Not A Slug"), "category"),
        (lambda m: m.__setitem__("provenance", {"doi": "not-a-doi"}), "DOI"),
        (
            lambda m: m.__setitem__("reproducibility", {"golden_trace_sha256": "xyz"}),
            "golden_trace",
        ),
    ],
)
def test_parse_rejects_invalid_controlled_fields(mutate, match) -> None:
    payload = _minimal_payload()
    mutate(payload)
    with pytest.raises(ModelDescriptorError, match=match):
        parse_model_descriptor(payload)


def test_completeness_tiers_rise_with_curation() -> None:
    """Each curation column lifts the descriptor to the next tier."""

    payload = _minimal_payload()
    payload["metadata"].update(  # type: ignore[union-attr]
        {"family": "Integrate-and-Fire", "category": "adaptive", "behavior_tags": ["tonic"]}
    )
    assert descriptor_completeness_tier(parse_model_descriptor(payload)) == 1

    payload["provenance"] = {"authors": ["Brette"], "year": 2005, "doi": "10.1152/jn.00686.2005"}
    payload["parameters"] = {
        "tau": {"default": 20.0, "unit": "ms", "range": [1.0, 100.0], "meaning": "time constant"}
    }
    payload["backends"] = {
        "python": {"status": "implemented"},
        "rust": {"status": "implemented", "parity": "ulp-bounded"},
    }
    assert descriptor_completeness_tier(parse_model_descriptor(payload)) == 2

    payload["reproducibility"] = {
        "reference_config": "golden/adex.json",
        "golden_trace_sha256": "a" * 64,
    }
    assert descriptor_completeness_tier(parse_model_descriptor(payload)) == 3


def test_generate_descriptor_reads_real_fields_and_provenance() -> None:
    """The generator reads code fields and carries over curated provenance."""

    payload = generate_descriptor_payload("AdExNeuron")
    assert payload["metadata"]["class_name"] == "AdExNeuron"
    # ``b`` is the adaptation increment parameter, not a state variable.
    assert "b" in payload["parameters"]
    assert list(payload["state"]) == ["v", "w"]
    # Non-numeric Literal fields (integrator choice) are not numeric parameters.
    assert "integrator" not in payload["parameters"]
    # Provenance carried over from the curated v1 schema.
    assert payload["provenance"]["doi"] == "10.1152/jn.00686.2005"
    # Curation-only fields are left empty, never fabricated.
    assert payload["metadata"]["family"] == ""
    assert all(p["unit"] == "" for p in payload["parameters"].values())


def test_generate_descriptor_handles_non_dataclass_model() -> None:
    """The generator introspects plain (non-dataclass) model constructors."""

    descriptor = generate_descriptor("HybridFisherPosnerLIFNeuron")
    param_names = {p.name for p in descriptor.parameters}
    assert "tau_m" in param_names
    assert descriptor.class_name == "HybridFisherPosnerLIFNeuron"


@pytest.mark.parametrize("class_name", sorted(_CLASS_TO_MODULE))
def test_every_model_generates_a_valid_descriptor(class_name: str) -> None:
    """Gate: every registered model yields a schema-valid descriptor whose
    parameter and state names are all real fields of the model (no invented
    names)."""

    descriptor = generate_descriptor(class_name)
    assert descriptor.class_name == class_name
    declared = {p.name for p in descriptor.parameters} | {s.name for s in descriptor.state}

    module = importlib.import_module(f"sc_neurocore.neurons.models.{_CLASS_TO_MODULE[class_name]}")
    cls = getattr(module, class_name)
    if dataclasses.is_dataclass(cls):
        real_fields = {f.name for f in dataclasses.fields(cls)}
    else:
        real_fields = set(inspect.signature(cls).parameters)
    # The synthetic fallback state "v" is allowed when a model declares none.
    invented = declared - real_fields - {"v"}
    assert invented == set(), f"{class_name}: descriptor names not in the model: {sorted(invented)}"
