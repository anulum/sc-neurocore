# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Refuse contradictory model-profile input at the public boundary

"""Malformed author input must surface a specific profile contradiction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from sc_neurocore.neurons.model_profile import ModelProfileError, parse_profile, resolve_profile
from sc_neurocore.neurons.universal_dsl import load_schema


@pytest.mark.parametrize(
    ("stem", "changes", "expected"),
    [
        ("lif", {"profile": "broken"}, "profile section must be a table"),
        ("lif", {"profile": {"contract": "unknown"}}, "profile contract"),
        ("lif", {"profile": {"unknown": True}}, "unknown profile keys"),
        ("lif", {"integration": {"dt": True}}, "positive finite number"),
        ("lif", {"integration": {"substeps": False}}, "positive integer"),
        ("lif", {"state": {"v": "invalid"}}, "non-numeric initial value"),
        ("lif", {"parameters": {"tau_m": "invalid"}}, "non-numeric default"),
        ("lif", {"profile": {"parameters": {"ghost": "source"}}}, "undeclared parameter"),
        ("lif", {"profile": {"refractory_register": "ghost"}}, "not a state variable"),
        ("lif", {"profile": {"refractory_register": "v"}}, "auxiliary register"),
        ("lif", {"profile": {"parameters": {"tau_m": "mystery"}}}, "parameter role"),
        ("lif", {"profile": {"admissible_methods": "euler"}}, "must be a list"),
        ("lif", {"profile": {"admissible_methods": ["rk4"]}}, "must include"),
        ("lif", {"extensions": {"integrator_options": ["map"]}}, "leave the ode family"),
        (
            "lif",
            {"profile": {"substep_kind": "invalid"}, "integration": {"substeps": 2}},
            "substep_kind",
        ),
        ("poisson", {"threshold": {"rng_seed": "bad"}}, "rng_seed must be an integer"),
    ],
)
def test_invalid_profile_input_is_reported(
    stem: str, changes: dict[str, Any], expected: str
) -> None:
    schema = deepcopy(load_schema(stem))
    for section, values in changes.items():
        if isinstance(values, dict) and isinstance(schema.get(section), dict):
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(schema[section].get(key), dict):
                    schema[section][key].update(value)
                else:
                    schema[section][key] = value
        else:
            schema[section] = values
    profile = resolve_profile(schema, stem=stem)
    assert any(expected in problem for problem in profile.problems), profile.problems


def test_public_profile_payload_refuses_bad_contract_and_missing_fields() -> None:
    payload = resolve_profile(load_schema("lif"), stem="lif").to_public_dict()
    wrong_contract = dict(payload, contract="foreign")
    with pytest.raises(ModelProfileError, match="profile payload contract"):
        parse_profile(wrong_contract)
    missing_science = dict(payload)
    del missing_science["scientific"]
    with pytest.raises(ModelProfileError, match="malformed profile payload"):
        parse_profile(missing_science)


def test_non_text_role_falls_back_to_biological_state() -> None:
    schema = deepcopy(load_schema("lif"))
    schema["profile"]["state"]["v"] = 42
    profile = resolve_profile(schema, stem="lif")
    assert profile.scientific.biological_state[0].name == "v"
    assert profile.numerical.auxiliary_registers == ()


def test_exact_relaxation_refuses_nonsmooth_self_dependence() -> None:
    schema = deepcopy(load_schema("sc_lapicque_lif"))
    schema["dynamics"]["v"] = "abs(v)"
    profile = resolve_profile(schema, stem="sc_lapicque_lif")
    assert any("not smooth in itself" in problem for problem in profile.problems)


def test_executable_method_without_dynamics_is_only_a_descriptive_record() -> None:
    schema = deepcopy(load_schema("lif"))
    schema["dynamics"] = {}
    profile = resolve_profile(schema, stem="lif")
    assert profile.realisation_kind == "descriptive-record"
    assert any("no dynamics" in problem for problem in profile.problems)


def test_declaring_a_derived_exactness_does_not_promote_the_model() -> None:
    schema = deepcopy(load_schema("lif"))
    schema["profile"]["exactness"] = "first-order"
    profile = resolve_profile(schema, stem="lif")
    assert profile.numerical.exactness == "first-order"
    assert profile.numerical.exactness_claimed is False
    assert profile.problems == ()


def test_stochastic_threshold_with_diffusion_noise_is_not_reproducible() -> None:
    schema = deepcopy(load_schema("escape_rate"))
    variable = next(iter(schema["dynamics"]))
    schema["dynamics"][variable] += " + xi"
    profile = resolve_profile(schema, stem="escape_rate")
    assert profile.numerical.randomness.kind == "lfsr16-threshold+diffusion-noise-global-rng"
    assert profile.numerical.randomness.reproducible is False


def test_poisson_detection_inside_an_ode_retains_the_reset_phase() -> None:
    schema = deepcopy(load_schema("lif"))
    schema["threshold"]["detection"] = "poisson"
    schema["threshold"]["probability_expression"] = "0.1"
    profile = resolve_profile(schema, stem="lif")
    assert profile.numerical.family == "ode"
    assert profile.numerical.evaluation_order == (
        "integrate",
        "probability",
        "lfsr-trial",
        "reset",
    )
