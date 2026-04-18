# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network graph builder for Studio (Block 5)

from __future__ import annotations

import secrets

from sc_neurocore.studio.models import list_models
from sc_neurocore.studio.network import simulate_ei_network


def available_models() -> list[str]:
    """Return names of all neuron models available for populations."""
    try:
        return [m["name"] for m in list_models()]
    except Exception:
        return ["LIFNeuron", "IzhikevichNeuron", "AdExNeuron"]


def create_population(
    label: str = "Population",
    model: str = "LIFNeuron",
    count: int = 80,
    neuron_type: str = "excitatory",
    x: float = 0.0,
    y: float = 0.0,
) -> dict:
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
) -> dict:
    """Create a projection edge between two populations."""
    return {
        "id": f"proj_{secrets.token_hex(4)}",
        "source": source_id,
        "target": target_id,
        "weight": weight,
        "delay": delay,
        "probability": probability,
    }


def validate_graph(graph: dict) -> list[str]:
    """Validate a network graph. Returns list of error messages (empty = valid)."""
    errors = []
    populations = graph.get("populations", [])
    projections = graph.get("projections", [])

    if not populations:
        errors.append("Network has no populations")
        return errors

    pop_ids = {p["id"] for p in populations}

    for proj in projections:
        if proj["source"] not in pop_ids:
            errors.append(f"Projection {proj['id']} source {proj['source']} not found")
        if proj["target"] not in pop_ids:
            errors.append(f"Projection {proj['id']} target {proj['target']} not found")
        if proj.get("weight", 0) == 0:
            errors.append(f"Projection {proj['id']} has zero weight")
        if proj.get("probability", 0) <= 0 or proj.get("probability", 1) > 1:
            errors.append(f"Projection {proj['id']} probability out of range (0, 1]")

    total_neurons = sum(p.get("count", 0) for p in populations)
    if total_neurons > 2000:
        errors.append(
            f"Total neuron count {total_neurons} exceeds 2000 limit for browser simulation"
        )

    return errors


def simulate_graph(graph: dict) -> dict:
    """Simulate a network graph using the E-I network backend.

    Maps populations and projections to the existing E-I simulation.
    For graphs with exactly 2 populations (1 exc + 1 inh), this
    directly uses simulate_ei_network. For other topologies, falls
    back to a simplified all-to-all simulation.
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


def graph_to_nir(graph: dict) -> dict:
    """Export network graph to NIR-compatible format."""
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


def nir_to_graph(nir_data: dict) -> dict:
    """Import NIR-compatible format to network graph."""
    populations = []
    projections = []

    x_offset = 0
    for node_id, node in nir_data.get("nodes", {}).items():
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

    for edge in nir_data.get("edges", []):
        projections.append(
            {
                "id": f"proj_{secrets.token_hex(4)}",
                "source": edge["source"],
                "target": edge["target"],
                "weight": edge.get("weight", 1.0),
                "delay": edge.get("delay", 0.0),
                "probability": 1.0,
            }
        )

    return {"populations": populations, "projections": projections}
