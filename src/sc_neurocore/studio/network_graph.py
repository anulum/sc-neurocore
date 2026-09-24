# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network graph builder for Studio (Block 5)

"""Network-canvas graph operations for Studio.

Population and projection factories, validation, simulation through the
public ``Network`` runtime (:mod:`sc_neurocore.studio.network_graph_spec`
resolves the graph, :mod:`sc_neurocore.studio.network_execution` lowers and
runs it) and the Studio graph envelope's JSON import/export. The explicit E-I template
(:func:`sc_neurocore.studio.network.simulate_ei_network`) is no longer used
for graphs: a graph runs the models, parameters, rules, signed weights, delays
and drives it declares, or is rejected with the field and reason.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from typing import Any

from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.model_introspection import _load_class
from sc_neurocore.studio.model_run_contract import (
    DT_OVERRIDE_REASON,
    ModelInputError,
    model_drive_contract,
    model_parameter_contracts,
)
from sc_neurocore.studio.models import list_models
from sc_neurocore.studio.network_execution import simulate_graph_spec
from sc_neurocore.studio.network_graph_spec import (
    DEFAULT_MODEL,
    GraphRejected,
    graph_issues,
    resolve_graph,
    validate_graph,
)


#: What this interchange actually is: a Studio network graph, whose node
#: ``type`` is a catalogue model name and whose edges carry a weight and a
#: delay. It is NOT the Neuromorphic Intermediate Representation. Earlier
#: exports named themselves ``nir`` at version ``0.1``, which was never true —
#: no NIR primitive is mapped here, and a real NIR file cannot be read by this
#: loader. Real NIR files are written and read by
#: :mod:`sc_neurocore.studio.network_nir`, which maps populations and
#: projections to NIR primitives through the same realisation the runtime uses.
GRAPH_ENVELOPE_FORMAT = "sc-neurocore.studio.network-graph"

#: Version of the honest envelope. The loader still accepts the legacy pair
#: below, so files exported before the rename keep opening.
#: Envelope version. `"2"` carries the population label and the connectivity
#: rule with its probability, seed and autapse decision; `"1"` carried none of
#: them, so a round trip through it silently became all-to-all. Both are read;
#: a `"1"` document keeps the documented defaults.
GRAPH_ENVELOPE_VERSION = "2"
LEGACY_GRAPH_ENVELOPE_VERSION = "1"
ACCEPTED_GRAPH_ENVELOPE_VERSIONS = frozenset(
    {GRAPH_ENVELOPE_VERSION, LEGACY_GRAPH_ENVELOPE_VERSION}
)

#: The envelope earlier exports wrote. Accepted on import, never written.
LEGACY_GRAPH_ENVELOPE_FORMAT = "nir"

#: Envelope names this loader accepts.
ACCEPTED_GRAPH_ENVELOPE_FORMATS = frozenset({GRAPH_ENVELOPE_FORMAT, LEGACY_GRAPH_ENVELOPE_FORMAT})


#: Contract version of the population model contract a canvas editor reads.
POPULATION_MODEL_CONTRACT_VERSION = "studio.population-model-contract.v1"


class ModelDiscoveryError(RuntimeError):
    """Raised when Studio model discovery cannot produce a trustworthy list."""


def population_model_admission(name: str) -> str | None:
    """Return why catalogue model ``name`` cannot form a population, or ``None``.

    A population is admissible when the model has a float drive ``step`` that
    the Studio protocol can satisfy and no ``seed`` constructor field (every
    neuron of a population would otherwise share the seed and its noise).
    """
    if name not in _CLASS_TO_MODULE:
        return "not a catalogue model"
    cls = _load_class(name)
    try:
        drive = model_drive_contract(name, cls)
    except ModelInputError as exc:
        return exc.reason
    if drive.kind == "int":
        return "integer-drive model: the public Network injects float currents"
    if "seed" in model_parameter_contracts(cls).overridable:
        return "seed field: every neuron of a population would share its noise"
    return None


def population_model_contract(name: str) -> dict[str, Any] | None:
    """Return the contract a population of model ``name`` is validated against.

    The canvas creates a population with a model's defaults and then has to let
    a user change them. It cannot do that honestly from a list of names: which
    constructor fields are numerically overridable, what kind each is, what it
    defaults to and *why* the others are not inputs are all decided by
    :mod:`sc_neurocore.studio.model_run_contract`, and a browser that guessed
    would be a second implementation of the contract, free to drift.

    Parameters
    ----------
    name : str
        Catalogue model name.

    Returns
    -------
    dict or None
        ``None`` when the model is not admissible for a population; otherwise
        ``schema_version``, ``model``, the ``parameters`` a population may
        override with their kind and default, the ``unsupported`` fields with
        the reason each is not an input, and the ``drive`` parameter the
        Studio protocol delivers the current through.
    """
    if population_model_admission(name) is not None:
        return None
    cls = _load_class(name)
    contracts = model_parameter_contracts(cls)
    drive = model_drive_contract(name, cls)
    return {
        "drive": {
            "kind": drive.kind,
            "parameter": drive.parameter,
            "positional_only": drive.positional_only,
        },
        "model": name,
        "parameters": [
            {
                "default": contract.default,
                "kind": contract.kind,
                "name": field,
            }
            for field, contract in sorted(contracts.overridable.items())
            # dt is overridable on the class and refused as an override by the
            # run contract, because the graph sets the timestep through its own
            # field. Offering it as a parameter would offer a field that is
            # always rejected.
            if field != "dt"
        ],
        "schema_version": POPULATION_MODEL_CONTRACT_VERSION,
        "unsupported": [
            {"name": field, "reason": reason}
            for field, reason in sorted(
                {**contracts.unsupported, "dt": DT_OVERRIDE_REASON}.items()
                if "dt" in contracts.overridable
                else contracts.unsupported.items()
            )
        ],
    }


def available_models() -> list[str]:
    """Return the names of the catalogue models admissible for populations.

    Raises
    ------
    ModelDiscoveryError
        When the catalogue yields no admissible model.
    """
    names = [m["name"] for m in list_models()]
    admitted = [name for name in names if population_model_admission(name) is None]
    if not admitted:
        raise ModelDiscoveryError("Studio model discovery returned no admissible models")
    return admitted


def create_population(
    label: str = "Population",
    model: str = DEFAULT_MODEL,
    count: int = 80,
    neuron_type: str = "excitatory",
    x: float = 0.0,
    y: float = 0.0,
    params: Mapping[str, float] | None = None,
    drive: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a population node for the network canvas.

    The node carries the fields the graph schema executes: catalogue ``model``,
    ``count``, ``neuron_type``, constructor ``params`` and the external
    ``drive`` (``{"kind": "none"}`` when omitted). Nothing is validated here;
    :func:`validate_graph` reports every problem of the assembled graph.
    """
    return {
        "id": f"pop_{secrets.token_hex(4)}",
        "type": "population",
        "label": label,
        "model": model,
        "count": count,
        "neuron_type": neuron_type,
        "position": {"x": x, "y": y},
        "params": dict(params or {}),
        "drive": dict(drive) if drive is not None else {"kind": "none"},
    }


def create_projection(
    source_id: str,
    target_id: str,
    weight: float = 0.1,
    delay: float = 0.0,
    probability: float = 0.2,
    rule: str = "random",
) -> dict[str, Any]:
    """Create a projection edge between two populations.

    ``weight`` is signed (negative for an inhibitory source), ``delay`` is in
    milliseconds and must be a whole number of graph timesteps, ``rule`` is
    ``random`` (with ``probability``) or ``all_to_all``.
    """
    edge: dict[str, Any] = {
        "id": f"proj_{secrets.token_hex(4)}",
        "source": source_id,
        "target": target_id,
        "weight": weight,
        "delay": delay,
        "rule": rule,
    }
    if rule == "random":
        edge["probability"] = probability
    return edge


def simulate_graph(graph: object) -> dict[str, Any]:
    """Simulate a network graph through the public ``Network`` runtime.

    Returns ``{"success": False, "errors": [...]}`` with every validation
    message when the graph cannot be resolved, otherwise the
    ``studio.network-graph-result.v1`` payload of
    :func:`sc_neurocore.studio.network_execution.simulate_graph_spec`.

    Raises
    ------
    GraphExecutionFailure
        When a resolved graph fails while running.
    """
    issues = graph_issues(graph)
    if issues:
        return {"success": False, "errors": [issue.message for issue in issues]}
    return simulate_graph_spec(resolve_graph(graph))


def graph_to_envelope(graph: object) -> dict[str, Any]:
    """Export a validated network graph as the Studio graph envelope (JSON).

    Raises
    ------
    ValueError
        When the graph does not validate.
    """
    if not isinstance(graph, Mapping):
        raise ValueError("Network graph must be an object")
    errors = validate_graph(graph)
    if errors:
        raise ValueError(f"Invalid network graph: {'; '.join(errors)}")

    nodes = {}
    edges = []

    for pop in graph.get("populations", []):
        node: dict[str, Any] = {
            "type": pop.get("model", DEFAULT_MODEL),
            "count": pop.get("count", 1),
            "neuron_type": pop.get("neuron_type", "excitatory"),
            "params": pop.get("params", {}),
        }
        # The label is what a reader named the population. Dropping it replaced
        # every name with its identifier on the way back in.
        label = pop.get("label")
        if isinstance(label, str) and label:
            node["label"] = label
        nodes[pop["id"]] = node

    for proj in graph.get("projections", []):
        edge: dict[str, Any] = {
            "source": proj["source"],
            "target": proj["target"],
            "weight": proj.get("weight", 1.0),
            "delay": proj.get("delay", 0.0),
        }
        # Connectivity is the topology, not decoration: without the rule and its
        # probability and seed, a random projection came back all-to-all — a
        # different network wearing the same weights.
        for field in ("rule", "probability", "seed", "autapses"):
            if field in proj:
                edge[field] = proj[field]
        edges.append(edge)

    return {
        "format": GRAPH_ENVELOPE_FORMAT,
        "version": GRAPH_ENVELOPE_VERSION,
        "nodes": nodes,
        "edges": edges,
    }


def envelope_to_graph(nir_data: object) -> dict[str, Any]:
    """Import a Studio graph envelope (JSON), current or legacy, to a network graph.

    Every node ``type`` must be a catalogue model name, and an unknown type is
    rejected rather than replaced by a default. Real NIR files are read by
    :func:`sc_neurocore.studio.network_nir.nir_file_to_graph`.

    A version-2 document carries the population label and each projection's
    connectivity rule with its probability, seed and autapse decision, so a
    round trip returns the network that was exported. A version-1 document
    carried none of those: its edges connect all-to-all, which is what it has
    always meant, and its populations are named by their identifiers.

    The assembled graph is validated against the graph schema with the Studio
    default timestep, so an import that would not execute (sign conflicts,
    delays that are not whole default steps, inadmissible models, budgets) is
    rejected here instead of surfacing later on the canvas.

    Raises
    ------
    ValueError
        On a malformed payload, a node type that is not a catalogue model, or
        an assembled graph that does not validate.
    """
    if not isinstance(nir_data, Mapping):
        raise ValueError("Network graph payload must be an object")
    declared = nir_data.get("format")
    if declared is not None and declared not in ACCEPTED_GRAPH_ENVELOPE_FORMATS:
        raise ValueError(
            f"unreadable interchange format {declared!r}: this loader reads the "
            f"Studio network graph envelope ({GRAPH_ENVELOPE_FORMAT!r}, or the "
            f"legacy {LEGACY_GRAPH_ENVELOPE_FORMAT!r} an earlier export wrote). "
            "It does not read the Neuromorphic Intermediate Representation; NIR "
            "files are imported as NIR, not as this envelope."
        )
    raw_nodes = nir_data.get("nodes", {})
    raw_edges = nir_data.get("edges", [])
    if not isinstance(raw_nodes, Mapping):
        raise ValueError("Graph envelope nodes must be an object")
    if not isinstance(raw_edges, list):
        raise ValueError("Graph envelope edges must be a list")

    populations = []
    projections = []

    x_offset = 0
    for node_id, node in raw_nodes.items():
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Graph envelope node ids must be non-empty strings")
        if not isinstance(node, Mapping):
            raise ValueError(f"Graph envelope node {node_id!r} must be an object")
        model = node.get("type", DEFAULT_MODEL)
        if not isinstance(model, str) or model not in _CLASS_TO_MODULE:
            raise ValueError(
                f"Graph envelope node {node_id!r} type {model!r} is not a catalogue model"
            )
        populations.append(
            {
                "id": node_id,
                "type": "population",
                "label": node_id,
                "model": model,
                "count": node.get("count", 1),
                "neuron_type": node.get("neuron_type", "excitatory"),
                "position": {"x": x_offset, "y": 0},
                "params": node.get("params", {}),
                "drive": {"kind": "none"},
                **(
                    {"label": node["label"]}
                    if isinstance(node.get("label"), str) and node["label"]
                    else {}
                ),
            }
        )
        x_offset += 200

    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, Mapping):
            raise ValueError(f"Graph envelope edge {index} must be an object")
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Graph envelope edge {index} source must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise ValueError(f"Graph envelope edge {index} target must be a non-empty string")
        projection: dict[str, Any] = {
            "id": f"proj_{secrets.token_hex(4)}",
            "source": source,
            "target": target,
            "weight": edge.get("weight", 1.0),
            "delay": edge.get("delay", 0.0),
            # A document that carries no rule is a version-1 export, which never
            # had one; all-to-all is what it has always meant and what its own
            # documentation promised.
            "rule": edge.get("rule", "all_to_all"),
        }
        for field in ("probability", "seed", "autapses"):
            if field in edge:
                projection[field] = edge[field]
        projections.append(projection)

    graph = {"populations": populations, "projections": projections}
    errors = validate_graph(graph)
    if errors:
        raise ValueError(f"Imported graph is not executable: {'; '.join(errors)}")
    return graph


__all__ = [
    "GraphRejected",
    "ModelDiscoveryError",
    "POPULATION_MODEL_CONTRACT_VERSION",
    "available_models",
    "create_population",
    "create_projection",
    "graph_issues",
    "envelope_to_graph",
    "graph_to_envelope",
    "population_model_admission",
    "population_model_contract",
    "simulate_graph",
    "validate_graph",
]
