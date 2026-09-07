# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model profile contract: separation, round-trip, representatives

"""The profile separates science from numerics and lowering without changing a number."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

import pytest

from sc_neurocore.neurons.model_profile import (
    EXECUTABLE_DETECTIONS,
    EXECUTABLE_METHODS,
    METHOD_TABLE,
    PROFILE_CONTRACT,
    ModelProfile,
    parse_profile,
    resolve_profile,
)
from sc_neurocore.neurons.schema_validator import validate_schema, validate_schema_dict
from sc_neurocore.neurons.universal_dsl import (
    UniversalNeuron,
    list_bundled_schemas,
    load_schema,
    schema_to_toml,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

REPRESENTATIVES = (
    "lapicque",
    "sc_lapicque_lif",
    "lif",
    "adex",
    "hodgkin_huxley",
    "wang_buzsaki",
    "exp_if",
    "rulkov_map",
    "chialvo_map",
    "escape_rate",
    "poisson",
    "coba_lif",
)


@pytest.fixture(scope="module")
def profiles() -> dict[str, ModelProfile]:
    """Resolve every bundled schema once."""
    return {name: resolve_profile(load_schema(name), stem=name) for name in list_bundled_schemas()}


def test_every_bundled_profile_round_trips_and_is_contradiction_free(
    profiles: dict[str, ModelProfile],
) -> None:
    """Payload → parse is the identity, and no bundled schema contradicts its profile."""
    for name, profile in profiles.items():
        payload = profile.to_public_dict()
        assert json.loads(json.dumps(payload)) == payload, name
        assert parse_profile(payload) == profile, name
        assert profile.contract == PROFILE_CONTRACT
        assert profile.problems == (), (name, profile.problems)
        assert profile.stem == name


def test_toml_and_json_twins_resolve_to_the_same_profile(profiles: dict[str, ModelProfile]) -> None:
    """The authored profile section survives both formats and the TOML export."""
    schema_dir = load_schema.__globals__["_SCHEMA_DIR"]
    for name in list_bundled_schemas():
        toml_profile = resolve_profile(load_schema(schema_dir / f"{name}.toml"), stem=name)
        json_profile = resolve_profile(load_schema(schema_dir / f"{name}.json"), stem=name)
        assert toml_profile == json_profile == profiles[name], name
        exported = tomllib.loads(schema_to_toml(load_schema(name)))
        assert resolve_profile(exported, stem=name) == profiles[name], name


def test_representatives_author_a_profile_and_the_rest_derive_one(
    profiles: dict[str, ModelProfile],
) -> None:
    """Twelve representatives carry an authored section; every other schema is derived."""
    authored = {name for name, profile in profiles.items() if profile.authored}
    assert authored == set(REPRESENTATIVES)
    for name in REPRESENTATIVES:
        assert profiles[name].is_executable, name
        assert profiles[name].numerical.time_unit in {"ms", "iteration"}, name


def test_executable_and_descriptive_records_are_partitioned_by_vocabulary(
    profiles: dict[str, ModelProfile],
) -> None:
    """A schema is executable iff its method and detection are in the runtime vocabulary."""
    for name, profile in profiles.items():
        schema = load_schema(name)
        method = schema["integration"].get("method", "euler")
        detection = schema.get("threshold", {}).get("detection", "level")
        expected = method in EXECUTABLE_METHODS and detection in EXECUTABLE_DETECTIONS
        assert profile.is_executable == expected, name
        if profile.is_executable:
            UniversalNeuron(schema)
        else:
            with pytest.raises(ValueError):
                UniversalNeuron(schema)
            assert profile.numerical.exactness == "not-executable"
            assert profile.numerical.macro_step is None
            assert profile.lowering.rtl_supported is False
            assert profile.lowering.limits
            warnings = [
                error
                for error in validate_schema(name)
                if error.level == "warning" and "descriptive record" in error.message
            ]
            assert warnings, name
    kinds = {profile.realisation_kind for profile in profiles.values()}
    assert kinds == {"executable", "descriptive-record"}
    assert sum(1 for p in profiles.values() if not p.is_executable) == 13


def test_lapicque_exact_flow_separates_event_latch_and_timebase(
    profiles: dict[str, ModelProfile],
) -> None:
    """Lapicque: sampled closed-form flow; excited is a register; dt is a timebase parameter."""
    profile = profiles["lapicque"]
    numerical = profile.numerical
    assert numerical.family == "map"
    assert numerical.exactness == "exact-flow"
    assert numerical.exactness_claimed is True
    assert [v.name for v in profile.scientific.biological_state] == ["v"]
    assert [r.name for r in numerical.auxiliary_registers] == ["excited"]
    assert numerical.auxiliary_registers[0].meaning == "latched first threshold attainment"
    assert [p.name for p in numerical.timebase_parameters] == ["dt"]
    assert numerical.timebase_parameters[0].unit == "ms"
    assert [p.name for p in profile.scientific.source_parameters] == [
        "v_threshold",
        "capacitance",
        "series_resistance",
        "polarization_resistance",
    ]
    assert numerical.admissible_methods == ("map",)
    assert numerical.evaluation_order == ("iterate", "level threshold", "reset")
    assert profile.scientific.biological_state[0].unit == "normalized polarization"


def test_sc_lapicque_lif_exponential_euler_is_verified_exact_and_stays_an_sc_identity(
    profiles: dict[str, ModelProfile],
) -> None:
    """The SC profile's exp_euler claim is checked symbolically; identity and defaults unchanged."""
    profile = profiles["sc_lapicque_lif"]
    assert profile.numerical.exactness == "exact-linear-relaxation"
    assert profile.numerical.exactness_claimed is True
    assert profile.scientific.name == "SCLapicqueLIFNeuron"
    assert profile.scientific.doi == ""
    assert profile.numerical.admissible_methods[0] == "exp_euler"
    assert set(profile.numerical.admissible_methods) == {
        "exp_euler",
        "euler",
        "rk4",
        "gauss_seidel",
    }
    schema = load_schema("sc_lapicque_lif")
    assert schema["parameters"] == {
        "v_rest": 0.0,
        "v_reset": 0.0,
        "v_threshold": 1.0,
        "tau": 20.0,
        "resistance": 1.0,
    }
    # Numerically: the exponential-Euler step equals the closed-form constant-current flow.
    neuron = UniversalNeuron(schema)
    tau, resistance, current = 20.0, 1.0, 0.5
    v = 0.0
    for _ in range(40):
        neuron.step(I=current)
        v = resistance * current + (v - resistance * current) * math.exp(-1.0 / tau)
        assert neuron.state["v"] == pytest.approx(v, abs=1e-12)


def test_euler_lif_is_first_order_and_cannot_claim_exactness(
    profiles: dict[str, ModelProfile],
) -> None:
    """Explicit Euler over the same linear membrane is a first-order realisation only."""
    assert profiles["lif"].numerical.exactness == "first-order"
    assert profiles["lif"].numerical.exactness_claimed is False
    schema = deepcopy(load_schema("lif"))
    schema["profile"]["exactness"] = "exact-flow"
    claimed = resolve_profile(schema, stem="lif")
    assert any("exactness claim 'exact-flow' is not admissible" in p for p in claimed.problems)
    errors = [e for e in validate_schema_dict(schema, "lif") if e.level == "error"]
    assert any("exactness claim" in e.message for e in errors)
    with pytest.raises(ValueError, match="contradicts its profile"):
        UniversalNeuron(schema)
    # Euler drifts from the exact flow by O(dt): the two realisations are not interchangeable.
    neuron = UniversalNeuron(load_schema("lif"))
    exact = -65.0
    for _ in range(10):
        neuron.step(I=10.0)
        exact = -65.0 + 10.0 + (exact - (-65.0 + 10.0)) * math.exp(-1.0 / 10.0)
    assert abs(neuron.state["v"] - exact) > 1e-3


def test_exp_euler_exactness_claim_is_refused_when_the_equation_is_not_affine() -> None:
    """A claim the symbolic check cannot confirm is a contradiction, not a label."""
    schema = deepcopy(load_schema("sc_lapicque_lif"))
    schema["dynamics"]["v"] = "(-(v - v_rest) ** 3 + resistance * I) / tau"
    claimed = resolve_profile(schema, stem="cubic")
    assert any("not affine in itself" in p for p in claimed.problems)
    coupled = deepcopy(load_schema("sc_lapicque_lif"))
    coupled["state"]["w"] = 0.0
    coupled["dynamics"] = {"v": "(-(v - v_rest) - w + resistance * I) / tau", "w": "-w / tau"}
    coupled["profile"]["state"]["w"] = "biological"
    profile = resolve_profile(coupled, stem="coupled")
    assert any("depends on other state" in p for p in profile.problems)


def test_rk4_time_subdivision_and_gauss_seidel_order(profiles: dict[str, ModelProfile]) -> None:
    """HH: 100 RK4 sub-steps of 0.01 ms make one 1 ms macro step; WB is sequential Euler."""
    hh = profiles["hodgkin_huxley"].numerical
    assert (hh.method, hh.exactness, hh.substeps, hh.substep_kind) == (
        "rk4",
        "fourth-order",
        100,
        "time-subdivision",
    )
    assert hh.macro_step == pytest.approx(1.0)
    assert hh.evaluation_order == (
        "integrate x100",
        "rising-edge threshold on the macro boundary",
        "no reset rule",
    )
    assert [p.name for p in hh.implementation_parameters] == ["v_threshold"]
    assert profiles["hodgkin_huxley"].lowering.rtl_supported is True
    assert profiles["hodgkin_huxley"].lowering.limits == (
        "no multiply pipelining with substeps > 1",
    )
    wb = profiles["wang_buzsaki"].numerical
    assert (wb.method, wb.exactness, wb.substeps, wb.macro_step) == (
        "gauss_seidel",
        "first-order-sequential",
        50,
        0.5,
    )
    # A sub-stepped resetting model cannot be lowered: the emitter refuses it.
    schema = deepcopy(load_schema("adex"))
    schema["integration"]["substeps"] = 4
    lowered = resolve_profile(schema, stem="adex-substeps")
    assert lowered.problems == ()
    assert lowered.lowering.rtl_supported is False
    assert "substeps > 1 lower only for crossing, non-resetting models" in lowered.lowering.limits


def test_published_maps_admit_only_the_recurrence(profiles: dict[str, ModelProfile]) -> None:
    """Rulkov and Chialvo are recurrences without a continuous timebase."""
    for name in ("rulkov_map", "chialvo_map"):
        numerical = profiles[name].numerical
        assert numerical.family == "map"
        assert numerical.exactness == "recurrence"
        assert numerical.admissible_methods == ("map",)
        assert numerical.time_unit == "iteration"
        assert numerical.timebase_parameters == ()
        assert numerical.evaluation_order[0] == "iterate"
    chialvo = profiles["chialvo_map"].numerical
    assert [p.name for p in chialvo.implementation_parameters] == ["x_threshold"]
    assert chialvo.implementation_parameters[0].meaning == "maintained observation convention"
    assert [p.name for p in profiles["rulkov_map"].scientific.source_parameters] == [
        "alpha",
        "sigma",
        "mu",
    ]


def test_stochastic_processes_state_their_randomness_contract(
    profiles: dict[str, ModelProfile],
) -> None:
    """Escape rate and Poisson draw only from the model-scoped seeded LFSR."""
    escape = profiles["escape_rate"].numerical
    assert escape.randomness.kind == "lfsr16-threshold"
    assert escape.randomness.seed == 44257
    assert escape.randomness.reproducible is True
    assert escape.exactness == "exact-linear-relaxation"
    assert escape.evaluation_order == ("integrate", "hazard = rate * dt", "lfsr-trial", "reset")
    assert escape.event.stochastic_expression.startswith("rho_0 * exp(")
    assert escape.event.condition == ""
    poisson = profiles["poisson"]
    assert poisson.numerical.family == "event-only"
    assert poisson.numerical.exactness == "event-only"
    assert poisson.numerical.evaluation_order == ("probability", "lfsr-trial")
    assert [p.name for p in poisson.numerical.timebase_parameters] == ["dt_ms"]
    assert poisson.numerical.timebase_parameters[0].unit == "ms"
    assert poisson.scientific.biological_state == ()
    deterministic = profiles["lif"].numerical.randomness
    assert (deterministic.kind, deterministic.seed, deterministic.reproducible) == (
        "none",
        None,
        True,
    )
    noisy = deepcopy(load_schema("lif"))
    noisy["dynamics"]["v"] = "-(v - v_rest) / tau_m + R * I / C + xi"
    assert resolve_profile(noisy, stem="noisy").numerical.randomness.kind == (
        "diffusion-noise-global-rng"
    )
    assert resolve_profile(noisy, stem="noisy").numerical.randomness.reproducible is False


def test_coba_stage_registers_are_lowering_state_not_biology(
    profiles: dict[str, ModelProfile],
) -> None:
    """COBA folds four RK4 stages under map: declared as stage iteration of one 0.1 ms step."""
    profile = profiles["coba_lif"]
    numerical = profile.numerical
    assert (numerical.method, numerical.substeps, numerical.substep_kind) == (
        "map",
        4,
        "stage-iteration",
    )
    assert numerical.macro_step == 0.1
    assert [v.name for v in profile.scientific.biological_state] == ["v", "g_e", "g_i"]
    assert {r.name for r in numerical.auxiliary_registers} == {
        "refractory_time",
        "spike_flag",
        "phase",
        "base_v",
        "base_ge",
        "base_gi",
        "last_k_v",
        "last_k_ge",
        "last_k_gi",
        "weighted_v",
        "weighted_ge",
        "weighted_gi",
    }
    assert numerical.event.refractory_register == "refractory_time"
    assert [p.name for p in numerical.implementation_parameters] == ["delta_ge", "delta_gi"]
    assert [p.name for p in numerical.timebase_parameters] == ["dt"]
    assert numerical.evaluation_order[0] == "stage x4"
    # Without the authored section the same schema is ambiguous and is refused.
    schema = deepcopy(load_schema("coba_lif"))
    del schema["profile"]
    derived = resolve_profile(schema, stem="coba_lif")
    assert any("must declare substep_kind" in p for p in derived.problems)
    with pytest.raises(ValueError, match="substep_kind"):
        UniversalNeuron(schema)
    # Stage iteration is a map notion: an ODE cannot declare it.
    ode = deepcopy(load_schema("hodgkin_huxley"))
    ode["profile"]["substep_kind"] = "stage-iteration"
    assert any("require method" in p for p in resolve_profile(ode, stem="hh").problems)


def test_authored_profile_contradictions_are_errors_in_the_static_validator() -> None:
    """Unknown quantities, timebase disagreement, foreign methods and bad roles are errors."""
    schema: dict[str, Any] = deepcopy(load_schema("lapicque"))
    schema["profile"]["state"]["ghost"] = "auxiliary"
    schema["profile"]["units"]["nothing"] = "mV"
    schema["profile"]["parameters"]["dt"] = "timebase"
    schema["parameters"]["dt"] = 0.02
    schema["profile"]["admissible_methods"] = ["map", "euler"]
    schema["profile"]["state"]["v"] = "sacred"
    profile = resolve_profile(schema, stem="lapicque")
    messages = "\n".join(profile.problems)
    assert "undeclared state variable 'ghost'" in messages
    assert "unknown quantity 'nothing'" in messages
    assert "timebase parameter 'dt' = 0.02 contradicts integration.dt = 0.01" in messages
    assert "leave the map family" in messages
    assert "state role 'sacred'" in messages
    errors = [e for e in validate_schema_dict(schema, "lapicque") if e.level == "error"]
    assert {e.section for e in errors} == {"profile"}
    assert len(errors) == len(profile.problems)
    unit_clash = deepcopy(load_schema("poisson"))
    unit_clash["profile"]["units"]["dt_ms"] = "s"
    assert any("time unit is 'ms'" in p for p in resolve_profile(unit_clash).problems)
    stray = deepcopy(load_schema("lif"))
    stray["profile"]["substep_kind"] = "time-subdivision"
    assert any("declared with substeps = 1" in p for p in resolve_profile(stray).problems)
    record = deepcopy(load_schema("hill_tononi"))
    record["profile"] = {"time_unit": "ms"}
    assert any("descriptive record cannot author" in p for p in resolve_profile(record).problems)


def test_a_dt_parameter_read_by_the_expressions_must_be_bound_to_the_timebase() -> None:
    """A parameter named dt that the equations read but that is not the step is a contradiction."""
    schema = deepcopy(load_schema("lapicque"))
    schema["profile"]["parameters"]["dt"] = "source"
    profile = resolve_profile(schema, stem="lapicque")
    assert any("not bound to integration.dt" in p for p in profile.problems)


def test_method_table_matches_the_derived_classes() -> None:
    """The published mapping table names every executable method exactly once."""
    assert sorted(row["method"] for row in METHOD_TABLE) == sorted(EXECUTABLE_METHODS)
    for row in METHOD_TABLE:
        schema = deepcopy(load_schema("lif"))
        if row["method"] == "map":
            schema = deepcopy(load_schema("rulkov_map"))
        else:
            schema["integration"]["method"] = row["method"]
            del schema["profile"]
        profile = resolve_profile(schema)
        assert profile.numerical.family == row["family"]
        assert profile.numerical.exactness == row["exactness"]


def test_a_model_the_studio_can_configure_for_rtl_declares_it_can_be_lowered() -> None:
    """No canonical compile schema may exist for a profile that refuses lowering.

    ``resolve_model_compile_configuration`` reads a model's canonical schema and
    never consults ``lowering.rtl_supported``. Today the two agree — the thirteen
    models whose profile refuses lowering all lack a canonical schema, so none is
    reachable — but nothing holds them together. Authoring a schema for one of
    them would silently make a model the Studio can configure for RTL out of a
    model that states it cannot be lowered.

    This binds the coincidence into a contract, which is cheaper than the guard
    the alternative would need: a refusal on a code path that cannot currently
    be reached is untestable apparatus.
    """
    from sc_neurocore.neurons.models import _CLASS_TO_MODULE
    from sc_neurocore.neurons.schema_module_aliases import schema_for_module
    from sc_neurocore.studio.models import get_model_detail

    contradictions = []
    for name, module in sorted(_CLASS_TO_MODULE.items()):
        try:
            stem = schema_for_module(module)
            profile = resolve_profile(load_schema(stem), stem=stem)
        except Exception:  # noqa: BLE001 - a model with no bundled schema is out of scope
            continue
        detail = get_model_detail(name) or {}
        if (
            isinstance(detail.get("compile_configuration"), dict)
            and not profile.lowering.rtl_supported
        ):
            contradictions.append(name)

    assert contradictions == [], (
        "these models carry a canonical compile schema while their profile declares "
        f"lowering.rtl_supported = false: {contradictions}"
    )
