# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model descriptor contract and generator tests

"""Tests for the declarative model descriptor (schema v2) and its generator."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path
import sys
import types
from typing import Any, Callable

import pytest

from sc_neurocore.neurons import universal_dsl
from sc_neurocore.neurons.descriptor_generator import (
    generate_descriptor,
    generate_descriptor_payload,
    merge_descriptor_payloads,
)
from sc_neurocore.neurons.model_descriptor import (
    MODEL_DESCRIPTOR_SCHEMA_VERSION,
    ModelDescriptorError,
    descriptor_completeness_tier,
    parse_model_descriptor,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE


def _minimal_payload() -> dict[str, Any]:
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
    del payload["metadata"]["class_name"]
    with pytest.raises(ModelDescriptorError, match="class_name"):
        parse_model_descriptor(payload)


def test_parse_rejects_unknown_schema_version() -> None:
    payload = _minimal_payload()
    payload["metadata"]["schema_version"] = 1
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
def test_parse_rejects_invalid_controlled_fields(
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    payload = _minimal_payload()
    mutate(payload)
    with pytest.raises(ModelDescriptorError, match=match):
        parse_model_descriptor(payload)


def test_completeness_tiers_rise_with_curation() -> None:
    """Each curation column lifts the descriptor to the next tier."""

    payload = _minimal_payload()
    payload["metadata"].update({"family": "Integrate-and-Fire", "category": "adaptive"})
    assert descriptor_completeness_tier(parse_model_descriptor(payload)) == 1

    # Tier 2 — scientifically curated: citeable provenance + every parameter curated.
    payload["provenance"] = {"authors": ["Brette"], "year": 2005, "doi": "10.1152/jn.00686.2005"}
    payload["parameters"] = {
        "tau": {"default": 20.0, "unit": "ms", "range": [1.0, 100.0], "meaning": "time constant"}
    }
    assert descriptor_completeness_tier(parse_model_descriptor(payload)) == 2

    # Tier 3 — engineering-verified: two implemented backends + a golden trace.
    payload["backends"] = {
        "python": {"status": "implemented"},
        "rust": {"status": "implemented", "parity": "ulp-bounded"},
    }
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
    # Family/category are filled from the curated taxonomy.
    assert payload["metadata"]["family"] == "Integrate-and-Fire"
    assert payload["metadata"]["category"] == "integrate-and-fire"
    # Curation-only fields with no source are left empty, never fabricated.
    assert payload["metadata"]["intended_use"] == []
    assert all(p["unit"] == "" for p in payload["parameters"].values())


def test_generate_descriptor_handles_non_dataclass_model() -> None:
    """The generator introspects plain (non-dataclass) model constructors."""

    descriptor = generate_descriptor("HybridFisherPosnerLIFNeuron")
    param_names = {p.name for p in descriptor.parameters}
    assert "tau_m" in param_names
    assert descriptor.class_name == "HybridFisherPosnerLIFNeuron"


@pytest.mark.parametrize(
    "class_name",
    ["", "_HiddenNeuron", "../AdExNeuron", "models/AdExNeuron", "AdExNeuron.toml"],
)
def test_generator_rejects_non_public_class_names_before_registry_lookup(
    class_name: str,
) -> None:
    """Descriptor generation fails closed before accepting path-like class names."""

    with pytest.raises(ValueError, match="public Python identifier"):
        generate_descriptor_payload(class_name)


def test_generator_reports_unknown_public_model_as_registry_miss() -> None:
    """Valid public identifiers that are absent from the registry remain registry misses."""

    with pytest.raises(KeyError, match="NotRegisteredNeuron"):
        generate_descriptor_payload("NotRegisteredNeuron")


def test_generator_preserves_missing_v1_schema_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model with no curated v1 schema still yields an honest generated descriptor."""

    def missing_schema(_source: str | Path) -> dict[str, Any]:
        raise FileNotFoundError("no curated schema")

    monkeypatch.setattr(universal_dsl, "load_schema", missing_schema)

    payload = generate_descriptor_payload("HybridFisherPosnerLIFNeuron")

    assert payload["metadata"]["class_name"] == "HybridFisherPosnerLIFNeuron"
    assert payload["provenance"] == {}


def test_generator_propagates_malformed_v1_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed curated schemas must fail the corpus refresh instead of being hidden."""

    def malformed_schema(_source: str | Path) -> dict[str, Any]:
        raise ValueError("unsupported curated schema")

    monkeypatch.setattr(universal_dsl, "load_schema", malformed_schema)

    with pytest.raises(ValueError, match="unsupported curated schema"):
        generate_descriptor_payload("AdExNeuron")


def test_generator_filters_plain_constructor_fields_from_registry_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain constructors keep numeric public defaults and skip non-parameters."""

    module_name = "sc_neurocore.neurons.models.synthetic_plain_descriptor"
    module = types.ModuleType(module_name)

    class SyntheticPlainNeuron:
        """Synthetic plain model for descriptor-generator contract coverage."""

        def __init__(
            self,
            tau: float = 1.0,
            *args: object,
            gain: float = 2.0,
            _hidden: float = 3.0,
            enabled: bool = True,
            label: str = "demo",
            **kwargs: object,
        ) -> None:
            self.tau = tau
            self.gain = gain
            self.args = args
            self.kwargs = kwargs
            self.enabled = enabled
            self.label = label

    module.__dict__["SyntheticPlainNeuron"] = SyntheticPlainNeuron
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(_CLASS_TO_MODULE, "SyntheticPlainNeuron", "synthetic_plain_descriptor")

    payload = generate_descriptor_payload("SyntheticPlainNeuron")

    assert payload["parameters"] == {
        "tau": {"default": 1.0, "unit": "", "meaning": ""},
        "gain": {"default": 2.0, "unit": "", "meaning": ""},
    }
    assert payload["state"] == {}


def test_generator_keeps_empty_structure_when_plain_signature_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uninspectable plain classes do not receive invented descriptor fields."""

    module_name = "sc_neurocore.neurons.models.synthetic_bad_signature"
    module = types.ModuleType(module_name)

    class BadSignatureNeuron:
        """Synthetic plain model whose constructor signature cannot be inspected."""

    type.__setattr__(BadSignatureNeuron, "__signature__", "not-a-signature")
    module.__dict__["BadSignatureNeuron"] = BadSignatureNeuron
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(_CLASS_TO_MODULE, "BadSignatureNeuron", "synthetic_bad_signature")

    payload = generate_descriptor_payload("BadSignatureNeuron")

    assert payload["parameters"] == {}
    assert payload["state"] == {}


def test_generator_treats_unavailable_source_as_no_dynamic_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-inspection failures leave state classification to field names."""

    def unavailable_source(_cls: type) -> str:
        raise OSError("source unavailable")

    monkeypatch.setattr(inspect, "getsource", unavailable_source)

    payload = generate_descriptor_payload("AdExNeuron")

    assert list(payload["state"]) == ["v", "w"]
    assert "b" in payload["parameters"]


def test_generator_handles_source_without_class_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source loader returning no class definition yields no dynamic-state hints."""

    def source_without_class(_cls: type) -> str:
        return "def not_a_model():\n    return None\n"

    monkeypatch.setattr(inspect, "getsource", source_without_class)

    payload = generate_descriptor_payload("AdExNeuron")

    assert list(payload["state"]) == ["v", "w"]
    assert "b" in payload["parameters"]


def test_merge_descriptor_payloads_preserves_curation_without_structural_drift() -> None:
    """The corpus merge keeps human curation without accepting stale structure."""

    regenerated = generate_descriptor_payload("AdExNeuron")
    curated: dict[str, Any] = {
        "metadata": {
            "schema_version": 999,
            "class_name": "WrongNeuron",
            "display_name": "Adaptive exponential IF",
            "summary": "Curated AdEx descriptor.",
            "maturity": "validated",
            "intended_use": ["adaptive-spiking-reference"],
        },
        "parameters": {
            "tau": {
                "default": -1.0,
                "unit": "ms",
                "range": [1.0, 100.0],
                "meaning": "membrane time constant",
            },
            "stale_parameter": {"unit": "arb"},
        },
        "state": {"v": {"init": 0.0, "unit": "mV", "meaning": "membrane potential"}},
        "provenance": {"authors": ["Brette", "Gerstner"], "year": 2005},
        "dynamics": {"v": "curated membrane equation"},
        "backends": {"python": {"status": "implemented"}, "rust": {"status": "implemented"}},
        "reproducibility": {"reference_config": "golden/adex.json"},
        "documentation": {"notes": "Preserved reviewer note."},
    }

    merged = merge_descriptor_payloads(curated, regenerated)

    assert merged["metadata"]["schema_version"] == MODEL_DESCRIPTOR_SCHEMA_VERSION
    assert merged["metadata"]["class_name"] == "AdExNeuron"
    assert merged["metadata"]["display_name"] == "Adaptive exponential IF"
    assert merged["metadata"]["summary"] == "Curated AdEx descriptor."
    assert merged["metadata"]["intended_use"] == ["adaptive-spiking-reference"]
    assert merged["parameters"]["tau"]["default"] == regenerated["parameters"]["tau"]["default"]
    assert merged["parameters"]["tau"]["unit"] == "ms"
    assert "stale_parameter" not in merged["parameters"]
    assert merged["state"]["v"]["init"] == regenerated["state"]["v"]["init"]
    assert merged["state"]["v"]["unit"] == "mV"
    assert merged["provenance"]["authors"] == ["Brette", "Gerstner"]
    assert merged["dynamics"]["v"] == "curated membrane equation"
    assert "rust" in merged["backends"]
    assert merged["reproducibility"]["reference_config"] == "golden/adex.json"
    assert merged["documentation"]["notes"] == "Preserved reviewer note."


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
