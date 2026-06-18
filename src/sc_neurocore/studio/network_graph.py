# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network graph builder for Studio (Block 5)

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
import secrets
from typing import Any

from sc_neurocore.studio.models import list_models
from sc_neurocore.studio.network import simulate_ei_network


class ModelDiscoveryError(RuntimeError):
    """Raised when Studio model discovery cannot produce a trustworthy list."""


def available_models() -> list[str]:
    """Return names of all neuron models available for populations."""
    names = [m["name"] for m in list_models()]
    if not names:
        raise ModelDiscoveryError("Studio model discovery returned no models")
    return names


def create_population(
    label: str = "Population",
    model: str = "LIFNeuron",
    count: int = 80,
    neuron_type: str = "excitatory",
    x: float = 0.0,
    y: float = 0.0,
) -> dict[str, Any]:
    """Create a population node for the network canvas."""
    return {
        "id": f"pop_{secrets.token_hex(4)}",
        "type": "population",
        "label": label,
        "model": model,
        "count": count,
        "neuron_type": neuron_type,
        "position": {"x": x, "y": y},
        "params": {},
    }


def create_projection(
    source_id: str,
    target_id: str,
    weight: float = 0.1,
    delay: float = 1.0,
    probability: float = 0.2,
) -> dict[str, Any]:
    """Create a projection edge between two populations."""
    return {
        "id": f"proj_{secrets.token_hex(4)}",
        "source": source_id,
        "target": target_id,
        "weight": weight,
        "delay": delay,
        "probability": probability,
    }


def validate_graph(graph: object) -> list[str]:
    """Validate a network graph. Returns list of error messages (empty = valid)."""
    errors = []
    if not isinstance(graph, Mapping):
        return ["Network graph must be an object"]

    populations = graph.get("populations", [])
    projections = graph.get("projections", [])

    if not isinstance(populations, list):
        errors.append("Network populations must be a list")
        populations = []
    if not isinstance(projections, list):
        errors.append("Network projections must be a list")
        projections = []

    if not populations:
        errors.append("Network has no populations")
        return errors

    pop_ids: set[str] = set()
    valid_populations: list[Mapping[str, Any]] = []
    for index, pop in enumerate(populations):
        if not isinstance(pop, Mapping):
            errors.append(f"Population {index} must be an object")
            continue
        pop_id = pop.get("id")
        if not isinstance(pop_id, str) or not pop_id:
            errors.append(f"Population {index} id must be a non-empty string")
            continue
        pop_ids.add(pop_id)
        valid_populations.append(pop)

    if not valid_populations:
        errors.append("Network has no valid populations")

    for index, proj in enumerate(projections):
        if not isinstance(proj, Mapping):
            errors.append(f"Projection {index} must be an object")
            continue
        proj_id = proj.get("id")
        proj_label = proj_id if isinstance(proj_id, str) and proj_id else str(index)
        source = proj.get("source")
        target = proj.get("target")
        if not isinstance(source, str) or not source:
            errors.append(f"Projection {proj_label} source must be a non-empty string")
        elif source not in pop_ids:
            errors.append(f"Projection {proj_label} source {source} not found")
        if not isinstance(target, str) or not target:
            errors.append(f"Projection {proj_label} target must be a non-empty string")
        elif target not in pop_ids:
            errors.append(f"Projection {proj_label} target {target} not found")
        weight = proj.get("weight", 0)
        if not isinstance(weight, Real) or isinstance(weight, bool):
            errors.append(f"Projection {proj_label} weight must be numeric")
        elif not math.isfinite(float(weight)):
            errors.append(f"Projection {proj_label} weight must be finite")
        elif float(weight) == 0:
            errors.append(f"Projection {proj_label} has zero weight")
        probability = proj.get("probability", 0)
        if not isinstance(probability, Real) or isinstance(probability, bool):
            errors.append(f"Projection {proj_label} probability must be numeric")
        elif not math.isfinite(float(probability)):
            errors.append(f"Projection {proj_label} probability must be finite")
        elif float(probability) <= 0 or float(probability) > 1:
            errors.append(f"Projection {proj_label} probability out of range (0, 1]")

    total_neurons = 0.0
    for index, pop in enumerate(valid_populations):
        count = pop.get("count", 0)
        if not isinstance(count, Real) or isinstance(count, bool):
            errors.append(f"Population {index} count must be numeric")
            continue
        if not math.isfinite(float(count)):
            errors.append(f"Population {index} count must be finite")
            continue
        total_neurons += float(count)
    if total_neurons > 2000:
        errors.append(
            f"Total neuron count {total_neurons:g} exceeds 2000 limit for browser simulation"
        )

    return errors


def simulate_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Simulate a network graph using the E-I network backend.

    Maps populations and projections to the existing E-I simulation.
    Only graphs with exactly 2 populations (1 exc + 1 inh) are
    currently supported; other topologies fail closed instead of
    being collapsed into an unfaithful surrogate.
    """
    populations = graph.get("populations", [])
    projections = graph.get("projections", [])
    duration = graph.get("duration", 200.0)
    dt = graph.get("dt", 0.1)

    errors = validate_graph(graph)
    if errors:
        return {"success": False, "errors": errors}

    exc_pops = [p for p in populations if p.get("neuron_type") == "excitatory"]
    inh_pops = [p for p in populations if p.get("neuron_type") == "inhibitory"]
    if len(exc_pops) != 1 or len(inh_pops) != 1 or len(populations) != 2:
        return {
            "success": False,
            "errors": [
                "Studio graph simulation currently requires exactly one excitatory "
                "and one inhibitory population; export richer topologies to NIR or "
                "run them through the full network backend."
            ],
        }

    n_exc = sum(p.get("count", 80) for p in exc_pops) if exc_pops else 80
    n_inh = sum(p.get("count", 20) for p in inh_pops) if inh_pops else 20

    # Extract weights from projections (use first matching or defaults)
    w_ee, w_ei, w_ie, w_ii = 0.1, 0.4, 0.1, 0.4
    p_conn = 0.2
    for proj in projections:
        src = next((p for p in populations if p["id"] == proj["source"]), None)
        tgt = next((p for p in populations if p["id"] == proj["target"]), None)
        if not src or not tgt:
            continue
        s_type = src.get("neuron_type", "excitatory")
        t_type = tgt.get("neuron_type", "excitatory")
        w = abs(proj.get("weight", 0.1))
        p_conn = proj.get("probability", 0.2)
        if s_type == "excitatory" and t_type == "excitatory":
            w_ee = w
        elif s_type == "excitatory" and t_type == "inhibitory":
            w_ie = w
        elif s_type == "inhibitory" and t_type == "excitatory":
            w_ei = w
        elif s_type == "inhibitory" and t_type == "inhibitory":
            w_ii = w

    result = simulate_ei_network(
        n_exc=n_exc,
        n_inh=n_inh,
        w_ee=w_ee,
        w_ei=w_ei,
        w_ie=w_ie,
        w_ii=w_ii,
        p_conn=p_conn,
        duration=duration,
        dt=dt,
    )
    result["success"] = True
    result["graph_summary"] = {
        "n_populations": len(populations),
        "n_projections": len(projections),
        "n_exc": n_exc,
        "n_inh": n_inh,
    }
    return result


def graph_to_nir(graph: object) -> dict[str, Any]:
    """Export network graph to NIR-compatible format."""
    if not isinstance(graph, Mapping):
        raise ValueError("Network graph must be an object")
    errors = validate_graph(graph)
    if errors:
        raise ValueError(f"Invalid network graph: {'; '.join(errors)}")

    nodes = {}
    edges = []

    for pop in graph.get("populations", []):
        nodes[pop["id"]] = {
            "type": "LIF" if "LIF" in pop.get("model", "LIF") else pop.get("model", "LIF"),
            "count": pop.get("count", 1),
            "neuron_type": pop.get("neuron_type", "excitatory"),
            "params": pop.get("params", {}),
        }

    for proj in graph.get("projections", []):
        edges.append(
            {
                "source": proj["source"],
                "target": proj["target"],
                "weight": proj.get("weight", 1.0),
                "delay": proj.get("delay", 0.0),
            }
        )

    return {
        "format": "nir",
        "version": "0.1",
        "nodes": nodes,
        "edges": edges,
    }


def nir_to_graph(nir_data: object) -> dict[str, Any]:
    """Import NIR-compatible format to network graph."""
    if not isinstance(nir_data, Mapping):
        raise ValueError("NIR payload must be an object")
    raw_nodes = nir_data.get("nodes", {})
    raw_edges = nir_data.get("edges", [])
    if not isinstance(raw_nodes, Mapping):
        raise ValueError("NIR nodes must be an object")
    if not isinstance(raw_edges, list):
        raise ValueError("NIR edges must be a list")

    populations = []
    projections = []

    x_offset = 0
    for node_id, node in raw_nodes.items():
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("NIR node ids must be non-empty strings")
        if not isinstance(node, Mapping):
            raise ValueError(f"NIR node {node_id!r} must be an object")
        populations.append(
            {
                "id": node_id,
                "type": "population",
                "label": node_id,
                "model": node.get("type", "LIFNeuron"),
                "count": node.get("count", 1),
                "neuron_type": node.get("neuron_type", "excitatory"),
                "position": {"x": x_offset, "y": 0},
                "params": node.get("params", {}),
            }
        )
        x_offset += 200

    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, Mapping):
            raise ValueError(f"NIR edge {index} must be an object")
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not source:
            raise ValueError(f"NIR edge {index} source must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise ValueError(f"NIR edge {index} target must be a non-empty string")
        projections.append(
            {
                "id": f"proj_{secrets.token_hex(4)}",
                "source": source,
                "target": target,
                "weight": edge.get("weight", 1.0),
                "delay": edge.get("delay", 0.0),
                "probability": 1.0,
            }
        )

    return {"populations": populations, "projections": projections}
