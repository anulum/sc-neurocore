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
    Silicon,
    Validation,
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


def test_parse_reads_doi_is_translation_flag() -> None:
    """A provenance section can mark its DOI as a translation of the cited work."""
    payload = _minimal_payload()
    payload["provenance"] = {
        "authors": ["Lapicque, L."],
        "year": 1907,
        "doi": "10.1007/s00422-007-0189-6",
        "doi_is_translation": True,
    }
    descriptor = parse_model_descriptor(payload)
    assert descriptor.provenance.doi_is_translation is True
    # The flag defaults to False when absent, so ordinary citations are unaffected.
    payload["provenance"].pop("doi_is_translation")
    assert parse_model_descriptor(payload).provenance.doi_is_translation is False


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


def test_completeness_tier_zero_for_descriptor_without_structure() -> None:
    """Descriptors with no parameters and no state remain at tier zero."""

    payload = _minimal_payload()
    payload["state"] = {}
    payload["parameters"] = {}

    descriptor = parse_model_descriptor(payload)

    assert descriptor.parameters == ()
    assert descriptor.state == ()
    assert descriptor_completeness_tier(descriptor) == 0


def test_parse_rejects_missing_required_metadata_section() -> None:
    """The metadata table is mandatory for every model descriptor."""

    payload = _minimal_payload()
    del payload["metadata"]

    with pytest.raises(ModelDescriptorError, match=r"missing the \[metadata\] section"):
        parse_model_descriptor(payload)


def test_parse_rejects_non_table_sections() -> None:
    """Section values must be TOML-style tables, not sequences or scalars."""

    payload = _minimal_payload()
    payload["state"] = ["v"]

    with pytest.raises(ModelDescriptorError, match=r"\[state\] must be a table"):
        parse_model_descriptor(payload)


def test_parse_rejects_string_tag_fields() -> None:
    """Discovery tag fields must be lists so scalar strings do not split silently."""

    payload = _minimal_payload()
    payload["metadata"]["intended_use"] = "simulation"

    with pytest.raises(ModelDescriptorError, match="tag fields"):
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


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m.__setitem__("integration", {"dt": True}), "expected a number"),
        (
            lambda m: m.__setitem__(
                "parameters",
                {"tau": {"default": 20.0, "range": [1.0]}},
            ),
            "range",
        ),
        (lambda m: m.__setitem__("provenance", {"authors": ["Brette"], "year": True}), "year"),
    ],
)
def test_parse_rejects_invalid_shape_and_numeric_fields(
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    """Malformed numeric and structured fields fail with descriptor errors."""

    payload = _minimal_payload()
    mutate(payload)

    with pytest.raises(ModelDescriptorError, match=match):
        parse_model_descriptor(payload)


def test_parse_accepts_legacy_scalar_and_mapping_forms() -> None:
    """Legacy scalar fields and compact backend forms normalize deterministically."""

    payload = _minimal_payload()
    payload["provenance"] = {"author": "Brette"}
    payload["state"] = {"v": -65.0}
    payload["parameters"] = {"tau": 20}
    payload["dynamics"] = {
        "v": {"expr": "(-v + input) / tau"},
        "w": {"expr": 3.0},
        "u": "du/dt",
    }
    payload["backends"] = {"python": "implemented"}

    descriptor = parse_model_descriptor(payload)

    assert descriptor.provenance.authors == ("Brette",)
    assert [(state.name, state.init) for state in descriptor.state] == [("v", -65.0)]
    assert [(parameter.name, parameter.default) for parameter in descriptor.parameters] == [
        ("tau", 20.0)
    ]
    assert descriptor.dynamics == {"v": "(-v + input) / tau", "w": "", "u": "du/dt"}
    assert [(backend.name, backend.status, backend.parity) for backend in descriptor.backends] == [
        ("python", "implemented", "n/a")
    ]


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


def test_validation_defaults_are_empty_and_unvalidated() -> None:
    """An absent [validation] section yields an empty, unvalidated facet."""
    descriptor = parse_model_descriptor(_minimal_payload())
    assert descriptor.validation == Validation()
    assert descriptor.validation.metric == "none"
    assert descriptor.validation.dynamics_faithful is False
    assert descriptor.validation.is_class_validated is False


def test_validation_is_class_validated_needs_metric_and_evidence() -> None:
    """The validated predicate requires both a non-trivial metric and evidence."""
    assert Validation(metric="parity", evidence="trace.json").is_class_validated is True
    assert Validation(metric="parity").is_class_validated is False
    assert Validation(metric="none", evidence="trace.json").is_class_validated is False


def test_parse_validation_section_reads_every_field() -> None:
    """The [validation] section round-trips its recorded evidence fields."""
    payload = _minimal_payload()
    payload["validation"] = {
        "dynamics_faithful": True,
        "metric": "statistical",
        "operating_point": "Poisson drive 20 Hz",
        "tolerance": "KS < 0.05",
        "evidence": "golden/adex_stats.json",
    }
    validation = parse_model_descriptor(payload).validation
    assert validation.dynamics_faithful is True
    assert validation.metric == "statistical"
    assert validation.operating_point == "Poisson drive 20 Hz"
    assert validation.tolerance == "KS < 0.05"
    assert validation.evidence == "golden/adex_stats.json"
    assert validation.is_class_validated is True


def test_parse_rejects_unknown_validation_metric() -> None:
    payload = _minimal_payload()
    payload["validation"] = {"metric": "vibes"}
    with pytest.raises(ModelDescriptorError, match="validation metric"):
        parse_model_descriptor(payload)


def test_parse_rejects_non_boolean_evidence_flag() -> None:
    payload = _minimal_payload()
    payload["validation"] = {"dynamics_faithful": "yes"}
    with pytest.raises(ModelDescriptorError, match="dynamics_faithful"):
        parse_model_descriptor(payload)


def test_silicon_defaults_are_empty() -> None:
    """An absent [silicon] section yields the below-H0 default facet."""
    descriptor = parse_model_descriptor(_minimal_payload())
    assert descriptor.silicon == Silicon()
    assert descriptor.silicon.compiles is False
    assert descriptor.silicon.clock_mhz is None
    assert descriptor.silicon.target_tier == ""


def test_parse_silicon_section_reads_every_field() -> None:
    """The [silicon] section carries the realisation ladder and its anchors."""
    payload = _minimal_payload()
    payload["silicon"] = {
        "compiles": True,
        "cosim_validated": True,
        "synthesised": True,
        "timing_closed": True,
        "formally_equivalent": True,
        "ppa_signed": True,
        "cosim_evidence": "cosim.log",
        "synth_report": "yosys.json",
        "timing_report": "sta.rpt",
        "equivalence_proof": "miter.smt2",
        "ppa_report": "openlane.json",
        "target_device": "xc7a35t",
        "clock_mhz": 125,
        "target_tier": "H3",
        "terminal_reason": "point neuron, deployable to H3",
    }
    silicon = parse_model_descriptor(payload).silicon
    assert silicon.compiles is True
    assert silicon.cosim_validated is True
    assert silicon.target_device == "xc7a35t"
    assert silicon.clock_mhz == pytest.approx(125.0)
    assert isinstance(silicon.clock_mhz, float)
    assert silicon.target_tier == "H3"
    assert silicon.terminal_reason == "point neuron, deployable to H3"


def test_parse_rejects_unknown_silicon_target_tier() -> None:
    payload = _minimal_payload()
    payload["silicon"] = {"target_tier": "H9"}
    with pytest.raises(ModelDescriptorError, match="target_tier"):
        parse_model_descriptor(payload)


def test_parse_rejects_non_numeric_clock() -> None:
    payload = _minimal_payload()
    payload["silicon"] = {"clock_mhz": "fast"}
    with pytest.raises(ModelDescriptorError, match="clock_mhz"):
        parse_model_descriptor(payload)


def test_parse_rejects_boolean_clock() -> None:
    payload = _minimal_payload()
    payload["silicon"] = {"clock_mhz": True}
    with pytest.raises(ModelDescriptorError, match="clock_mhz"):
        parse_model_descriptor(payload)


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
            "name": "Curated AdEx Descriptive Name",
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
        "validation": {
            "dynamics_faithful": True,
            "metric": "parity",
            "operating_point": "schema-DSL cosim Q16.16",
            "tolerance": "class-correct spike-count band",
            "evidence": "tests/test_cosimulation.py::TestQ1616Precision::test_adex_q1616_parity",
        },
        "silicon": {
            "compiles": True,
            "cosim_validated": True,
            "cosim_evidence": "tests/test_cosimulation.py::TestQ1616Precision::test_adex_q1616_parity",
            "target_tier": "H1",
            "terminal_reason": "point-neuron SC/RTL path; signed PPA out of scope",
        },
        "documentation": {
            "notes": "Preserved reviewer note.",
            "slug": "models/curated_adex_slug",
        },
    }

    merged = merge_descriptor_payloads(curated, regenerated)

    assert merged["metadata"]["schema_version"] == MODEL_DESCRIPTOR_SCHEMA_VERSION
    assert merged["metadata"]["class_name"] == "AdExNeuron"
    # The curated descriptive name and documentation slug are authoritative
    # overlays: a hand-written name/slug is never overwritten by the generic
    # generator default derived from the class/module name.
    assert merged["metadata"]["name"] == "Curated AdEx Descriptive Name"
    assert merged["metadata"]["name"] != regenerated["metadata"]["name"]
    assert merged["documentation"]["slug"] == "models/curated_adex_slug"
    assert merged["documentation"]["slug"] != regenerated["documentation"]["slug"]
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
    assert merged["validation"]["dynamics_faithful"] is True
    assert merged["validation"]["metric"] == "parity"
    assert merged["validation"]["evidence"].endswith("test_adex_q1616_parity")
    assert merged["silicon"]["compiles"] is True
    assert merged["silicon"]["cosim_validated"] is True
    assert merged["silicon"]["target_tier"] == "H1"


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
