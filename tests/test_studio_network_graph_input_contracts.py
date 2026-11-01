# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — public network graph input contracts

"""Validate malformed graph requests before constructing an executable network."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest

from sc_neurocore.studio.network_graph import create_population, create_projection
from sc_neurocore.studio.network_graph_spec import (
    GraphExecutionMode,
    GraphRejected,
    graph_issues,
    resolve_graph,
)


def _graph() -> dict[str, Any]:
    """Return a small valid request made through the public canvas constructors."""
    source = create_population(label="Source", count=2)
    target = create_population(label="Target", count=2)
    source["id"], target["id"] = "source", "target"
    projection = create_projection("source", "target", weight=1.0, probability=0.5)
    projection["id"] = "projection"
    return {
        "populations": [source, target],
        "projections": [projection],
        "duration": 1.0,
        "dt": 0.1,
        "seed": 7,
    }


def _refuses(graph: dict[str, Any], field: str) -> None:
    """Require a located public issue and failed resolution without request mutation."""
    before = deepcopy(graph)
    issues = graph_issues(graph)
    assert field in {issue.field for issue in issues}
    with pytest.raises(GraphRejected) as refusal:
        resolve_graph(graph)
    assert refusal.value.field == issues[0].field
    assert refusal.value.reason == issues[0].message
    assert graph == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", None),
        ("duration", -1),
        ("duration", 10**400),
        ("dt", None),
        ("dt", -1),
        ("seed", None),
    ],
)
def test_timing_and_seed_fields_are_located_before_execution(field: str, value: object) -> None:
    """Invalid clocks, overflowing integers and missing seeds keep their request field."""
    graph = _graph()
    graph[field] = value
    _refuses(graph, field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", None),
        ("id", ""),
        ("type", "projection"),
        ("label", 12),
        ("neuron_type", None),
        ("params", []),
        ("model", None),
    ],
)
def test_population_fields_refuse_malformed_values(field: str, value: object) -> None:
    """Population structure and model parameters are validated at their own fields."""
    graph = _graph()
    graph["populations"][0][field] = value
    _refuses(graph, f"populations[0].{field}")


@pytest.mark.parametrize(
    ("drive", "field"),
    [
        (12, "drive"),
        ({"kind": "constant", "current": 1.0, "weight": 1.0}, "drive"),
        ({"kind": "poisson", "rate_hz": 10.0, "weight": 1.0, "seed": -1}, "drive.seed"),
        ({"kind": "poisson", "rate_hz": 10.0, "weight": 1.0, "seed": 2**32}, "drive.seed"),
        ({"kind": "poisson", "rate_hz": 10.0, "weight": 1.0, "seed": False}, "drive.seed"),
    ],
)
def test_population_drives_refuse_shape_conflicts_and_invalid_seeds(
    drive: object, field: str
) -> None:
    """Drive declarations reject conflicting fields and invalid random streams."""
    graph = _graph()
    graph["populations"][0]["drive"] = drive
    _refuses(graph, "populations[0]." + field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", None),
        ("source", None),
        ("target", None),
        ("source", "missing"),
        ("weight", "1"),
        ("weight", float("inf")),
        ("weight", 0),
        ("probability", "0.5"),
        ("seed", -1),
        ("seed", 2**32),
        ("seed", True),
        ("autapses", 1),
    ],
)
def test_projection_fields_refuse_malformed_values(field: str, value: object) -> None:
    """An invalid endpoint, numeric value or random stream cannot form a projection."""
    graph = _graph()
    graph["projections"][0][field] = value
    _refuses(graph, f"projections[0].{field}")


@pytest.mark.parametrize("collection", ["populations", "projections"])
def test_graph_members_must_be_objects(collection: str) -> None:
    """A malformed collection member is refused without losing its position."""
    graph = _graph()
    graph[collection][0] = None
    _refuses(graph, collection + "[0]")


def test_bad_graph_seed_does_not_start_population_or_projection_streams() -> None:
    """A missing graph seed refuses even otherwise valid derived Poisson streams."""
    graph = _graph()
    graph["seed"] = None
    graph["populations"][0]["drive"] = {"kind": "poisson", "rate_hz": 10.0, "weight": 1.0}
    _refuses(graph, "seed")


def test_invalid_population_id_cannot_leave_an_executable_empty_network() -> None:
    """A graph with no usable population IDs reports the missing valid population."""
    graph = _graph()
    graph["populations"] = [{"id": ""}]
    graph["projections"] = []
    _refuses(graph, "populations")


def test_constructor_rejects_inconsistent_population_parameters() -> None:
    """Valid individual fields still must satisfy the real neuron's joint contract."""
    graph = _graph()
    graph["populations"][0]["params"] = {"v_rest": 2.0, "v_threshold": 1.0}
    _refuses(graph, "populations[0].params")


def test_missing_drive_and_null_parameters_keep_the_constructor_defaults() -> None:
    """Omitted input and null overrides resolve to the public model's real defaults."""
    graph = _graph()
    del graph["populations"][0]["drive"]
    graph["populations"][0]["params"] = None
    spec = resolve_graph(graph)
    assert spec.population("source").drive.kind == "none"
    assert spec.population("target") is spec.populations[1]
    assert spec.population("source").inputs.constructor_kwargs == {"dt": 0.1}
    with pytest.raises(KeyError, match="missing"):
        spec.population("missing")


@pytest.mark.parametrize("overflow", ["duration", "delay"])
def test_finite_values_with_an_overflowing_step_ratio_are_refused(overflow: str) -> None:
    """Finite scalar inputs cannot overflow the effective step or delay calculation."""
    graph = _graph()
    graph["dt"] = 1e-308
    graph["duration"] = 1e308 if overflow == "duration" else 1e-308
    if overflow == "delay":
        graph["projections"][0]["delay"] = 1e308
    _refuses(graph, "duration" if overflow == "duration" else "projections[0].delay")


def test_unknown_server_execution_profile_cannot_select_a_work_budget() -> None:
    """A runtime-invalid profile is rejected before admitting any graph work."""
    graph = _graph()
    mode = cast(GraphExecutionMode, "unregistered")
    assert [issue.field for issue in graph_issues(graph, execution_mode=mode)] == ["execution_mode"]
    with pytest.raises(GraphRejected, match="execution_mode"):
        resolve_graph(graph, execution_mode=mode)


def test_async_specification_records_its_server_budget_in_the_public_digest() -> None:
    """The async profile preserves topology while declaring its distinct work ceiling."""
    graph = _graph()
    sync = resolve_graph(graph).to_public_dict()
    asynchronous = resolve_graph(graph, execution_mode="async_job").to_public_dict()
    assert asynchronous["execution_mode"] == "async_job"
    assert asynchronous["limits"]["max_neuron_steps"] == 2_000_000
    assert sync["limits"]["max_neuron_steps"] == 1_000_000
    assert asynchronous["populations"] == sync["populations"]
    assert asynchronous["projections"] == sync["projections"]
    assert asynchronous["graph_sha256"] != sync["graph_sha256"]
