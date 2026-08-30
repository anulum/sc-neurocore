# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (generator) from former test_model_descriptor.py

from __future__ import annotations

from tests.model_descriptor_support import *  # noqa: F403


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


@pytest.mark.parametrize(
    ("class_name", "method", "dt"),
    [
        ("GLMNeuron", "map", 1.0),
        ("ParallelSpikingNeuron", "map", 1.0),
        ("TwoCompartmentLIFNeuron", "map", 1.0),
        ("SCResettingParallelSpikingNeuron", "map", 1.0),
        ("SCExponentialTwoCompartmentLIFNeuron", "map", 1.0),
        ("SCLeakyTwoCompartmentLIFNeuron", "euler", 1.0),
    ],
)
def test_schema_less_models_keep_declared_integration_contracts(
    class_name: str, method: str, dt: float
) -> None:
    payload = generate_descriptor_payload(class_name)
    assert payload["integration"] == {"dt": dt, "method": method}


def test_schema_less_compatibility_parameters_are_not_reclassified_as_state() -> None:
    exponential = generate_descriptor_payload("SCExponentialTwoCompartmentLIFNeuron")
    leaky = generate_descriptor_payload("SCLeakyTwoCompartmentLIFNeuron")
    resetting = generate_descriptor_payload("SCResettingParallelSpikingNeuron")
    for payload in (exponential, leaky):
        assert "theta" in payload["parameters"]
        assert "dt" in payload["parameters"]
        assert "theta" not in payload["state"]
    assert "kernel_size" in resetting["parameters"]
    assert "kernel_size" not in resetting["state"]


@pytest.mark.parametrize(
    ("source_class", "compatibility_class", "source_parameters", "compatibility_parameters"),
    [
        (
            "LapicqueNeuron",
            "SCLapicqueLIFNeuron",
            {"v_threshold", "capacitance", "series_resistance", "polarization_resistance"},
            {"v_rest", "v_reset", "v_threshold", "tau", "resistance"},
        ),
        (
            "QuadraticIFNeuron",
            "SCSymmetricQuadraticIFNeuron",
            {"v_reset", "v_peak"},
            {"v_reset", "v_peak"},
        ),
    ],
)
def test_source_and_sc_compatibility_descriptors_keep_distinct_profiles(
    source_class: str,
    compatibility_class: str,
    source_parameters: set[str],
    compatibility_parameters: set[str],
) -> None:
    """Count-bearing source descriptors never inherit inactive SC profile fields."""
    source = generate_descriptor_payload(source_class)
    compatibility = generate_descriptor_payload(compatibility_class)

    assert set(source["parameters"]) == source_parameters
    assert set(compatibility["parameters"]) == compatibility_parameters


def test_quadratic_if_source_descriptor_uses_latham_numeric_profile() -> None:
    """The count-bearing QIF descriptor follows the named Latham source constructor."""
    source = generate_descriptor_payload("QuadraticIFNeuron")
    compatibility = generate_descriptor_payload("SCSymmetricQuadraticIFNeuron")

    assert source["parameters"]["v_reset"]["default"] == -3.0
    assert source["parameters"]["v_peak"]["default"] == pytest.approx(31.0 / 3.0)
    assert source["integration"] == {
        "dt": 0.05,
        "method": "exact_held_current_riccati_flow",
    }
    assert compatibility["integration"] == {
        "dt": 0.01,
        "method": "exact_held_current_riccati_flow",
    }
