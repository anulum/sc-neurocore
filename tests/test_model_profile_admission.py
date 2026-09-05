# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Profile admission in the real consumers

"""Contradictory or unsupported overrides are refused by every consumer of the profile."""

from __future__ import annotations

import math
from copy import deepcopy

import pytest

from sc_neurocore.neurons.model_profile import (
    ProfileAdmissionError,
    admit_overrides,
    resolve_profile,
)
from sc_neurocore.neurons.reference_traces import (
    ReferenceTraceProtocol,
    ReferenceTraceProvenance,
    ReferenceTraceSpec,
    load_reference_trace_spec,
    simulate_reference_trace,
    validate_reference_trace,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import resolve_model_compile_configuration


def test_map_method_override_is_refused_and_ode_alternatives_are_derived() -> None:
    """A published map is never integrated as an ODE; an ODE may switch integrator."""
    with pytest.raises(ProfileAdmissionError, match="not admissible for the map profile"):
        UniversalNeuron.from_schema("rulkov_map", method_override="euler")
    with pytest.raises(ProfileAdmissionError, match="not admissible for the ode profile"):
        UniversalNeuron.from_schema("lif", method_override="map")
    neuron = UniversalNeuron.from_schema("lif", method_override="exp_euler")
    realised = neuron.realised_profile()
    assert realised["declared_method"] == "euler"
    assert realised["method"] == "exp_euler"
    assert realised["exactness"] == "linearised-exponential"
    assert realised["derived"] is True
    assert UniversalNeuron.from_schema("lif").realised_profile()["derived"] is False


def test_recurrence_without_timebase_refuses_a_step_override() -> None:
    """dt has no meaning for an iteration-indexed map; equality is still admitted."""
    with pytest.raises(ProfileAdmissionError, match="no continuous timebase"):
        UniversalNeuron.from_schema("chialvo_map", dt_override=0.5)
    neuron = UniversalNeuron.from_schema("chialvo_map", dt_override=1.0)
    assert neuron.admitted_overrides.dt == 1.0
    assert neuron.admitted_overrides.derived is False


def test_timebase_parameters_follow_the_step_override() -> None:
    """Lapicque's dt parameter and Poisson's dt_ms move with the overridden step."""
    lapicque = UniversalNeuron.from_schema("lapicque", dt_override=0.5)
    assert lapicque.admitted_overrides.parameters == {"dt": 0.5}
    assert lapicque.to_equation_neuron().parameters["dt"] == 0.5
    assert lapicque.to_equation_neuron().dt == 0.5
    # The sampled exact flow at a coarser step equals the closed form at that step.
    v = 0.0
    r, rho, k, source = 10.0, 1.0, 1.1, 2.0
    v_inf = source * rho / (r + rho)
    beta = k * r * rho / (r + rho)
    for _ in range(5):
        lapicque.step(I=source)
        v = v_inf + (v - v_inf) * math.exp(-0.5 / beta)
        assert lapicque.state["v"] == pytest.approx(v, abs=1e-12)
    poisson = UniversalNeuron.from_schema("poisson", dt_override=2.0, rng_seed_override=7)
    assert poisson.to_equation_neuron().parameters["dt_ms"] == 2.0
    assert poisson.admitted_overrides.rng_seed == 7
    with pytest.raises(ProfileAdmissionError, match="contradicts dt"):
        UniversalNeuron.from_schema("poisson", dt_override=1.0, parameter_overrides={"dt_ms": 2.0})
    with pytest.raises(ProfileAdmissionError, match="contradicts dt"):
        UniversalNeuron.from_schema("lapicque", parameter_overrides={"dt": 0.02})
    same = UniversalNeuron.from_schema("lapicque", parameter_overrides={"dt": 0.01})
    assert same.admitted_overrides.parameters == {"dt": 0.01}


def test_seed_override_is_refused_for_a_deterministic_profile() -> None:
    """A seed that nothing consumes is a contradiction, not a silently ignored knob."""
    with pytest.raises(ProfileAdmissionError, match="draws no randomness"):
        UniversalNeuron.from_schema("lif", rng_seed_override=3)
    seeded = UniversalNeuron.from_schema("escape_rate", rng_seed_override=3)
    assert seeded.to_equation_neuron().stochastic_rng_initial_seed == 3
    assert seeded.realised_profile()["rng_seed"] == 3
    assert UniversalNeuron.from_schema("escape_rate").realised_profile()["rng_seed"] == 44257


def test_descriptive_records_and_contradictory_schemas_are_refused_at_construction() -> None:
    """Neither a record outside the vocabulary nor a contradicted profile runs."""
    with pytest.raises(ValueError, match="descriptive record"):
        admit_overrides(resolve_profile(load_schema("hill_tononi"), stem="hill_tononi"))
    with pytest.raises(ValueError):
        UniversalNeuron.from_schema("hill_tononi")
    schema = deepcopy(load_schema("lapicque"))
    schema["parameters"]["dt"] = 0.02
    with pytest.raises(ProfileAdmissionError, match="contradicts integration.dt"):
        UniversalNeuron(schema)
    bad_dt = deepcopy(load_schema("lif"))
    with pytest.raises(ProfileAdmissionError, match="finite and positive"):
        UniversalNeuron(bad_dt, dt_override=0.0)


def test_studio_compile_admits_only_the_profile_family() -> None:
    """Studio integrator options stay inside the family; the evidence carries the profile."""
    detail = get_model_detail("RulkovMapNeuron")
    assert detail is not None
    assert detail["compile_configuration"]["integrators"] == ["map"]
    contract = detail["profile_contract"]
    assert contract["contract"] == "sc-neurocore.model-profile.v1"
    assert contract["numerical"]["admissible_methods"] == ["map"]
    with pytest.raises(ValueError, match="not declared"):
        resolve_model_compile_configuration({"model_name": "RulkovMapNeuron", "integrator": "rk4"})
    adex = get_model_detail("AdExNeuron")
    assert adex is not None
    assert adex["compile_configuration"]["integrators"] == ["euler", "rk4"]
    assert adex["profile_contract"]["numerical"]["family"] == "ode"
    configuration = resolve_model_compile_configuration(
        {"model_name": "AdExNeuron", "integrator": "rk4", "q_format": "Q16.16"}
    )
    profile = configuration.to_public_dict()["profile"]
    assert isinstance(profile, dict)
    assert profile["declared_method"] == "euler"
    assert profile["method"] == "rk4"
    assert profile["exactness"] == "fourth-order"
    assert profile["derived"] is True
    # A detail that declares a foreign integrator cannot smuggle it past the profile.
    forged = deepcopy(detail)
    forged["compile_configuration"]["integrators"] = ["map", "rk4"]
    with pytest.raises(ValueError, match="leaves the numerical family"):
        resolve_model_compile_configuration(
            {"model_name": "RulkovMapNeuron", "integrator": "rk4"},
            detail_getter=lambda _name: forged,
        )


def test_studio_detail_names_the_canonical_profile_per_class() -> None:
    """Two classes sharing a module get their own profile, not the module's."""
    lapicque = get_model_detail("LapicqueNeuron")
    sc_lapicque = get_model_detail("SCLapicqueLIFNeuron")
    assert lapicque is not None and sc_lapicque is not None
    assert lapicque["profile_contract"]["stem"] == "lapicque"
    assert sc_lapicque["profile_contract"]["stem"] == "sc_lapicque_lif"
    assert lapicque["profile_contract"]["numerical"]["exactness"] == "exact-flow"
    assert sc_lapicque["profile_contract"]["numerical"]["exactness"] == "exact-linear-relaxation"
    record = get_model_detail("HillTononiNeuron")
    assert record is not None
    assert record["compile_configuration"] is None
    assert record["profile_contract"]["realisation_kind"] == "descriptive-record"


def test_reference_trace_protocol_dt_is_admitted_through_the_profile() -> None:
    """A committed protocol runs under its schema's timebase; a foreign dt is refused."""
    report = validate_reference_trace("ermentrout_kopell_theta_euler_doi")
    assert report.passed
    spec = load_reference_trace_spec("ermentrout_kopell_theta_euler_doi")
    assert spec.protocol.dt == 0.1
    foreign = ReferenceTraceSpec(
        name="ek-foreign-dt",
        schema_name=spec.schema_name,
        runner=spec.runner,
        protocol=ReferenceTraceProtocol(
            dt=1.0,
            steps=spec.protocol.steps,
            inputs=spec.protocol.inputs,
            state_variables=spec.protocol.state_variables,
        ),
        provenance=ReferenceTraceProvenance(kind="test", source="test", equation="test"),
        expected_features=spec.expected_features,
        tolerances=spec.tolerances,
    )
    with pytest.raises(ProfileAdmissionError, match="no continuous timebase"):
        simulate_reference_trace(foreign)
    # An ODE protocol may sample a different step: the two closed-form traces still pass.
    for name in (
        "lif_constant_current_closed_form",
        "sc_lapicque_lif_constant_current_closed_form",
    ):
        assert validate_reference_trace(name).passed, name
