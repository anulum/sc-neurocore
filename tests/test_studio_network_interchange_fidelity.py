# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from typing import Any

from sc_neurocore.studio.network_graph import (
    GRAPH_ENVELOPE_VERSION,
    LEGACY_GRAPH_ENVELOPE_VERSION,
    graph_to_nir,
    nir_to_graph,
)


def _graph() -> dict[str, Any]:
    """A two-population network whose connectivity is random, not all-to-all."""
    return {
        "populations": [
            {
                "id": "p1",
                "label": "Excitatory LIF pool",
                "model": "LapicqueNeuron",
                "count": 80,
                "neuron_type": "excitatory",
                "position": {"x": 0, "y": 0},
                "params": {},
            },
            {
                "id": "p2",
                "label": "Inhibitory",
                "model": "AdExNeuron",
                "count": 20,
                "neuron_type": "inhibitory",
                "position": {"x": 200, "y": 0},
                "params": {},
            },
        ],
        "projections": [
            {
                "id": "e1",
                "source": "p1",
                "target": "p2",
                "weight": 0.5,
                "rule": "random",
                "probability": 0.137,
                "seed": 7,
                "autapses": False,
            }
        ],
    }


def test_a_round_trip_returns_the_network_that_was_exported() -> None:
    """Connectivity is the topology; losing it returns a different network.

    Version 1 carried neither the rule nor its probability, so a `random`
    projection came back `all_to_all` — the same weights over a different graph,
    with nothing to notice it by.
    """
    restored = nir_to_graph(graph_to_nir(_graph()))
    projection = restored["projections"][0]

    assert projection["rule"] == "random"
    assert projection["probability"] == 0.137
    assert projection["seed"] == 7
    assert projection["autapses"] is False


def test_a_population_keeps_the_name_a_reader_gave_it() -> None:
    """Every label was replaced by its identifier, not only names holding LIF."""
    restored = nir_to_graph(graph_to_nir(_graph()))

    assert [p["label"] for p in restored["populations"]] == [
        "Excitatory LIF pool",
        "Inhibitory",
    ]


def test_the_export_declares_the_version_that_carries_connectivity() -> None:
    """A reader must be able to tell a faithful document from a lossy one."""
    exported = graph_to_nir(_graph())

    assert exported["version"] == GRAPH_ENVELOPE_VERSION
    assert GRAPH_ENVELOPE_VERSION != LEGACY_GRAPH_ENVELOPE_VERSION


def test_a_version_one_document_still_imports_with_its_own_meaning() -> None:
    """An older export carried no rule, and all-to-all is what it always meant.

    Reading it as anything else would invent a topology the document never
    stated.
    """
    legacy = {
        "format": "sc-neurocore.studio.network-graph",
        "version": LEGACY_GRAPH_ENVELOPE_VERSION,
        "nodes": {
            "p1": {"type": "LapicqueNeuron", "count": 4, "neuron_type": "excitatory", "params": {}},
            "p2": {"type": "AdExNeuron", "count": 4, "neuron_type": "inhibitory", "params": {}},
        },
        "edges": [{"source": "p1", "target": "p2", "weight": 0.5, "delay": 0.0}],
    }

    restored = nir_to_graph(legacy)
    projection = restored["projections"][0]

    assert projection["rule"] == "all_to_all"
    assert "probability" not in projection
    assert [p["label"] for p in restored["populations"]] == ["p1", "p2"]
