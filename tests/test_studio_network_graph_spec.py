# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio network graph schema and resolution

"""Fail-closed resolution of a Studio network graph into its executable spec."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.studio.network_graph import (
    available_models,
    create_population,
    create_projection,
    population_model_admission,
)
from sc_neurocore.studio.network_graph_spec import (
    DEFAULT_MODEL,
    DEFAULT_SEED,
    GRAPH_SPEC_SCHEMA_VERSION,
    MAX_NEURON_STEPS,
    MAX_NEURONS,
    GraphRejected,
    derived_seed,
    graph_issues,
    resolve_graph,
    validate_graph,
)


def _graph(**overrides: Any) -> dict[str, Any]:
    exc = create_population(label="E", count=40, drive={"kind": "constant", "current": 1.2})
    exc["id"] = "e"
    inh = create_population(label="I", count=10, neuron_type="inhibitory")
    inh["id"] = "i"
    ei = create_projection("e", "i", weight=40.0, delay=0.2, probability=0.3)
    ei["id"] = "ei"
    ie = create_projection("i", "e", weight=-40.0, rule="all_to_all")
    ie["id"] = "ie"
    graph: dict[str, Any] = {
        "populations": [exc, inh],
        "projections": [ei, ie],
        "duration": 50.0,
        "dt": 0.1,
    }
    graph.update(overrides)
    return graph


def _only_error(graph: dict[str, Any], fragment: str) -> None:
    errors = validate_graph(graph)
    assert len(errors) == 1, errors
    assert fragment in errors[0], errors


class TestResolution:
    def test_default_graph_resolves_with_effective_parameters_and_exact_steps(self) -> None:
        spec = resolve_graph(_graph())
        assert spec.n_steps == 500
        assert spec.n_neurons == 50
        assert spec.neuron_steps == 25_000
        exc, inh = spec.populations
        assert exc.model == DEFAULT_MODEL
        assert exc.inputs.dt == 0.1
        assert exc.inputs.dt_source == "override"
        assert exc.inputs.constructor_kwargs == {"dt": 0.1}
        assert exc.offset == 0 and inh.offset == 40
        assert exc.drive.kind == "constant" and exc.drive.current == 1.2
        assert inh.drive.kind == "none"
        ei, ie = spec.projections
        assert ei.delay_steps == 2 and ei.delay_ms == 0.2
        assert ei.sign == "excitatory" and ie.sign == "inhibitory"
        assert ie.rule == "all_to_all" and ie.probability == 1.0
        assert ei.seed == derived_seed(DEFAULT_SEED, 1, 0)
        assert ie.seed == derived_seed(DEFAULT_SEED, 1, 1)
        assert ei.seed != ie.seed
        assert ei.seed_source == "derived"

    def test_public_dict_is_path_free_and_digest_bound(self) -> None:
        first = resolve_graph(_graph()).to_public_dict()
        second = resolve_graph(_graph()).to_public_dict()
        assert first["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
        assert first["graph_sha256"] == second["graph_sha256"]
        assert first["populations"][0]["parameters"]["tau"] == 20.0
        assert first["projections"][0]["delay_steps"] == 2
        assert "/" not in str(first)
        changed = resolve_graph(_graph(seed=7)).to_public_dict()
        assert changed["graph_sha256"] != first["graph_sha256"]
        assert changed["projections"][0]["seed"] == derived_seed(7, 1, 0)

    def test_moving_a_node_leaves_the_run_and_its_digest_unchanged(self) -> None:
        """Layout travels with a saved graph but is not part of what runs."""
        placed = _graph()
        moved = _graph()
        placed["populations"][0]["position"] = {"x": 0, "y": 0}
        moved["populations"][0]["position"] = {"x": 640.5, "y": -1200.0}
        moved["populations"][1]["position"] = {"x": 3.0, "y": 9000.0}

        assert resolve_graph(moved).to_public_dict() == resolve_graph(placed).to_public_dict()

    def test_explicit_seeds_and_params_are_taken_as_given(self) -> None:
        graph = _graph()
        graph["projections"][0]["seed"] = 123
        graph["populations"][0]["params"] = {"tau": 10.0, "v_threshold": 2.0}
        graph["populations"][1]["drive"] = {
            "kind": "poisson",
            "rate_hz": 50.0,
            "weight": 2.0,
            "seed": 9,
        }
        spec = resolve_graph(graph)
        assert spec.projections[0].seed == 123
        assert spec.projections[0].seed_source == "request"
        assert spec.populations[0].inputs.constructor_kwargs == {
            "tau": 10.0,
            "v_threshold": 2.0,
            "dt": 0.1,
        }
        assert spec.populations[1].drive.seed == 9
        assert spec.populations[1].drive.seed_source == "request"
        graph["populations"][1]["drive"].pop("seed")
        derived = resolve_graph(graph).populations[1].drive
        assert derived.seed == derived_seed(DEFAULT_SEED, 2, 1)
        assert derived.seed_source == "derived"

    def test_resolve_raises_with_the_first_field(self) -> None:
        graph = _graph()
        graph["populations"][0]["count"] = 2.5
        graph["projections"][1]["weight"] = 3.0
        with pytest.raises(GraphRejected) as info:
            resolve_graph(graph)
        assert info.value.field == "populations[0].count"
        assert info.value.to_public_detail()["error"] == "graph_rejected"
        assert len(graph_issues(graph)) == 2


class TestRejections:
    def test_unknown_model_and_missing_default_model_name(self) -> None:
        graph = _graph()
        graph["populations"][0]["model"] = "LIFNeuron"
        _only_error(graph, "model 'LIFNeuron' is not a catalogue model")

    @pytest.mark.parametrize("count", [0, -3, 2.5, True, "40", None])
    def test_count_must_be_a_positive_integer(self, count: object) -> None:
        graph = _graph()
        graph["populations"][0]["count"] = count
        errors = validate_graph(graph)
        assert len(errors) == 1
        assert "count must be" in errors[0]

    def test_duplicate_population_and_projection_ids(self) -> None:
        graph = _graph()
        graph["populations"][1]["id"] = "e"
        errors = validate_graph(graph)
        assert any("Population e id is duplicated" in e for e in errors)
        graph = _graph()
        graph["projections"][1]["id"] = "ei"
        _only_error(graph, "Projection ei id is duplicated")

    def test_missing_endpoint(self) -> None:
        graph = _graph()
        graph["projections"][0]["target"] = "ghost"
        _only_error(graph, "target ghost not found")

    def test_sign_must_agree_with_the_source_population(self) -> None:
        graph = _graph()
        graph["projections"][1]["weight"] = 40.0
        _only_error(graph, "inhibitory sources need a negative weight")
        graph = _graph()
        graph["projections"][0]["weight"] = -40.0
        _only_error(graph, "excitatory sources need a positive weight")

    def test_conflicting_rules(self) -> None:
        graph = _graph()
        graph["projections"][1]["probability"] = 0.2
        _only_error(graph, "rule all_to_all conflicts with probability 0.2")
        graph = _graph()
        del graph["projections"][0]["probability"]
        _only_error(graph, "rule random needs an explicit probability")
        graph = _graph()
        graph["projections"][0]["rule"] = "ring"
        errors = validate_graph(graph)
        assert any("rule must be one of" in e for e in errors)

    @pytest.mark.parametrize("probability", [0.0, 1.5, -0.1, float("nan")])
    def test_probability_domain(self, probability: float) -> None:
        graph = _graph()
        graph["projections"][0]["probability"] = probability
        errors = validate_graph(graph)
        assert len(errors) == 1 and "probability" in errors[0]

    def test_delay_must_be_whole_steps(self) -> None:
        graph = _graph()
        graph["projections"][0]["delay"] = 0.25
        _only_error(graph, "delay 0.25 ms is not a whole number of 0.1 ms steps")
        graph = _graph(dt=0.05)
        graph["projections"][0]["delay"] = 0.25
        assert resolve_graph(graph).projections[0].delay_steps == 5
        graph = _graph()
        graph["projections"][0]["delay"] = -1.0
        _only_error(graph, "delay must be a finite non-negative number")

    def test_autapses_only_on_self_projections(self) -> None:
        graph = _graph()
        graph["projections"][0]["autapses"] = True
        _only_error(graph, "autapses only apply to a self-projection")
        loop = create_projection("e", "e", weight=1.0, probability=0.5)
        loop["autapses"] = True
        graph = _graph()
        graph["projections"].append(loop)
        assert resolve_graph(graph).projections[2].autapses is True

    def test_unknown_fields_are_rejected_not_ignored(self) -> None:
        graph = _graph()
        graph["projections"][0]["delay_ms"] = 1.0
        _only_error(graph, "Projection ei has unknown fields: delay_ms")
        graph = _graph()
        graph["populations"][0]["tau"] = 5.0
        _only_error(graph, "Population e has unknown fields: tau")
        graph = _graph(p_conn=0.2)
        _only_error(graph, "Network graph has unknown fields: p_conn")

    def test_model_contract_errors_name_the_population_field(self) -> None:
        graph = _graph()
        graph["populations"][0]["params"] = {"tau_m": 20.0}
        issues = graph_issues(graph)
        assert len(issues) == 1
        assert issues[0].field == "populations[0].params.tau_m"
        assert "unknown parameter" in issues[0].message
        graph = _graph()
        graph["populations"][0]["model"] = "AkidaNeuron"
        _only_error(graph, "integer-drive model")
        graph = _graph()
        graph["populations"][0]["model"] = "StochasticLIFNeuron"
        _only_error(graph, "perfectly correlated noise")

    def test_fixed_step_model_rejects_the_graph_timestep(self) -> None:
        graph = _graph()
        graph["populations"][0]["model"] = "IntegerQIFNeuron"
        errors = validate_graph(graph)
        assert len(errors) == 1
        assert "dt: model has a fixed step of 1.0 ms" in errors[0]

    def test_drive_blocks(self) -> None:
        graph = _graph()
        graph["populations"][0]["drive"] = {"kind": "constant"}
        _only_error(graph, "constant drive needs a finite current")
        graph = _graph()
        graph["populations"][0]["drive"] = {"kind": "poisson", "rate_hz": 0.0, "weight": 0.0}
        errors = validate_graph(graph)
        assert len(errors) == 2
        graph = _graph()
        graph["populations"][0]["drive"] = {"kind": "none", "current": 1.0}
        _only_error(graph, "drive of kind none carries fields: current")
        graph = _graph()
        graph["populations"][0]["drive"] = {"kind": "ramp"}
        _only_error(graph, "drive kind must be one of")

    def test_budgets_refuse_instead_of_shortening(self) -> None:
        graph = _graph(duration=10.0)
        graph["populations"][0]["count"] = MAX_NEURONS
        _only_error(graph, f"exceeds {MAX_NEURONS} limit")
        graph = _graph(duration=0.05)
        _only_error(graph, "yields no complete step")
        graph = _graph(duration=MAX_NEURON_STEPS / 50 * 0.1 + 1.0)
        _only_error(graph, "neuron-steps, above the synchronous limit")
        graph = _graph(dt=1e-4, duration=1000.0)
        errors = validate_graph(graph)
        assert any("exceed the synchronous limit" in e for e in errors)

    @pytest.mark.parametrize("seed", [-1, 2**32, 1.5, "42"])
    def test_seed_domain(self, seed: object) -> None:
        graph = _graph(seed=seed)
        _only_error(graph, "seed must be an integer in [0, 2^32)")

    def test_all_issues_are_reported_together(self) -> None:
        graph = _graph()
        graph["populations"][0]["count"] = 0
        graph["projections"][0]["delay"] = 0.33
        graph["projections"][1]["weight"] = 1.0
        assert len(validate_graph(graph)) == 3

    def test_non_object_and_empty_graphs(self) -> None:
        assert validate_graph([]) == ["Network graph must be an object"]
        assert validate_graph({"populations": [], "projections": []}) == [
            "Network has no populations"
        ]
        errors = validate_graph({"populations": "x", "projections": {}})
        assert "Network populations must be a list" in errors[0]


class TestAdmission:
    def test_available_models_are_the_admissible_catalogue_models(self) -> None:
        models = available_models()
        assert DEFAULT_MODEL in models
        assert "AdExNeuron" in models
        assert "AkidaNeuron" not in models
        assert "StochasticLIFNeuron" not in models
        assert len(models) > 100

    def test_population_model_admission_reasons(self) -> None:
        assert population_model_admission(DEFAULT_MODEL) is None
        assert population_model_admission("LIFNeuron") == "not a catalogue model"
        assert "integer-drive" in str(population_model_admission("AkidaNeuron"))
        assert "seed field" in str(population_model_admission("StochasticLIFNeuron"))
