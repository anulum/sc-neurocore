# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused neuron descriptor defensive-contract coverage

"""Close defensive neuron, descriptor, and trace-loader behavior branches."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sc_neurocore.neurons import descriptor_generator
from sc_neurocore.neurons._stochastic_threshold import (
    Lfsr16Threshold,
    lfsr16_advance,
    probability_to_lfsr16_threshold,
)
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.model_descriptor import (
    ModelDescriptorError,
    Reproducibility,
    parse_model_descriptor,
)
from sc_neurocore.neurons.reference_trace_io import _load_spec_file
from tests.model_descriptor_support import _minimal_payload


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (lambda: lfsr16_advance(True), "state"),
        (lambda: probability_to_lfsr16_threshold(math.nan), "probability"),
        (lambda: Lfsr16Threshold().restore(0), "state"),
    ],
)
def test_stochastic_threshold_rejects_invalid_direct_contracts(
    operation: object, match: str
) -> None:
    """Direct primitive calls reject invalid state and probability values."""

    with pytest.raises(ValueError, match=match):
        operation()  # type: ignore[operator]


@pytest.mark.parametrize(
    "overrides",
    [
        {"detection": "level", "rate_expression": "1.0"},
        {"detection": "level", "probability_expression": "0.5"},
        {
            "detection": "escape_rate",
            "rate_expression": "1.0",
            "probability_expression": "0.5",
        },
        {"detection": "poisson"},
        {"detection": "poisson", "probability_expression": "0.5", "rate_expression": "1.0"},
    ],
)
def test_equation_neuron_rejects_mismatched_stochastic_expressions(
    overrides: dict[str, object],
) -> None:
    """Detection modes accept only their own required stochastic expression."""

    with pytest.raises(ValueError):
        EquationNeuron(equations={"v": "0.0"}, state={"v": 0.0}, **overrides)


def test_equation_neuron_rejects_escape_detection_without_rate() -> None:
    """Escape-rate detection requires an explicit rate expression."""

    with pytest.raises(ValueError, match="requires rate_expression"):
        EquationNeuron(equations={"v": "0.0"}, detection="escape_rate")


def test_equation_neuron_rejects_nonpositive_stochastic_timestep() -> None:
    """Stochastic detection requires a finite positive integration timestep."""

    with pytest.raises(ValueError, match="dt must be finite and positive"):
        EquationNeuron(
            equations={"v": "0.0"},
            detection="poisson",
            probability_expression="0.5",
            dt=0.0,
        )


def test_equation_neuron_rolls_back_state_when_integration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed stochastic integration restores its macro-boundary state."""

    neuron = EquationNeuron(
        equations={"v": "0.0"},
        state={"v": 3.0},
        detection="poisson",
        probability_expression="0.5",
    )

    def fail_integrator(**_kwargs: float) -> None:
        neuron.state["v"] = 99.0
        raise RuntimeError("integration failed")

    monkeypatch.setattr(neuron, "_integrate_once", fail_integrator)

    with pytest.raises(RuntimeError, match="integration failed"):
        neuron.step()
    assert neuron.state == {"v": 3.0}


def test_equation_neuron_rejects_missing_stochastic_runtime() -> None:
    """A corrupted stochastic runtime fails before consuming probability."""

    neuron = EquationNeuron(
        equations={"v": "0.0"},
        detection="poisson",
        probability_expression="0.5",
    )
    neuron._stochastic_rng = None

    with pytest.raises(RuntimeError, match="runtime was not initialised"):
        neuron.step()


@pytest.mark.parametrize(
    ("detection", "expression", "dt", "match"),
    [
        ("escape_rate", "-1.0", 0.1, "escape rate"),
        ("escape_rate", "1e15", 2.0, "escape hazard"),
        ("poisson", "2.0", 0.1, "spike probability"),
    ],
)
def test_equation_neuron_rejects_invalid_stochastic_runtime_values(
    detection: str, expression: str, dt: float, match: str
) -> None:
    """Runtime rates, hazards, and probabilities remain finite and bounded."""

    keyword = "rate_expression" if detection == "escape_rate" else "probability_expression"
    neuron = EquationNeuron(
        equations={"v": "0.0"},
        detection=detection,
        dt=dt,
        **{keyword: expression},
    )
    if match == "escape hazard":
        neuron.dt = 1.0e308

    with pytest.raises(FloatingPointError, match=match):
        neuron.step()


def test_descriptor_helpers_fail_closed_on_uninspectable_classes() -> None:
    """Constructor and source-introspection failures never fabricate state."""

    class RequiresArgument:
        def __init__(self, required: float) -> None:
            self.v = required

    assert descriptor_generator._instance_state_init(RequiresArgument, "v") is None
    assert descriptor_generator._is_mirror_field(int, "v") is False


def test_descriptor_generation_skips_curated_runtime_and_property_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Curated parameters take precedence over inferred runtime state fields."""

    class SyntheticNeuron:
        @property
        def rng_state(self) -> int:
            return 7

        def step(self) -> None:
            self.v = 1.0

    monkeypatch.setitem(descriptor_generator._CLASS_TO_MODULE, "SyntheticNeuron", "synthetic")
    monkeypatch.setattr(descriptor_generator, "_load_class", lambda _name: SyntheticNeuron)
    monkeypatch.setattr(
        descriptor_generator,
        "_load_v1_schema",
        lambda _module: {"parameters": {"v": 1.0, "rng_state": 7.0}},
    )

    payload = descriptor_generator.generate_descriptor_payload("SyntheticNeuron")

    assert payload["state"] == {}


def test_descriptor_merge_preserves_curation_only_runtime_state() -> None:
    """A curated runtime-only state survives structural regeneration."""

    merged = descriptor_generator.merge_descriptor_payloads(
        {"state": {"v": {"init": -65.0, "unit": "mV"}}},
        {"metadata": {}, "parameters": {}, "state": {}},
    )

    assert merged["state"]["v"] == {"init": -65.0, "unit": "mV"}


def test_reproducibility_returns_variants_without_a_primary_digest() -> None:
    """The typed value object preserves variants for explicit diagnostics."""

    variants = ("b" * 64,)
    assert Reproducibility(golden_trace_sha256_variants=variants).golden_trace_digests == variants


def test_descriptor_parser_rejects_variants_without_primary_digest() -> None:
    """Parsed descriptors require a primary digest before measured variants."""

    payload = _minimal_payload()
    payload["reproducibility"] = {"golden_trace_sha256_variants": ["b" * 64]}

    with pytest.raises(ModelDescriptorError, match="require a primary digest"):
        parse_model_descriptor(payload)


def test_reference_trace_loader_rejects_unknown_runner(tmp_path: Path) -> None:
    """Corpus loading rejects a runner before dispatching its payload parser."""

    path = tmp_path / "unknown-runner.json"
    path.write_text(json.dumps({"model": {"runner": "unknown"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="runner 'unknown' is not supported"):
        _load_spec_file(path)
