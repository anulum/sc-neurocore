# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio graph envelope fidelity

"""Preserve Studio network meaning through JSON graph envelopes and execution."""

from __future__ import annotations

from typing import Any
from copy import deepcopy
import json

import pytest

from sc_neurocore.studio.network_graph import (
    GRAPH_ENVELOPE_VERSION,
    LEGACY_GRAPH_ENVELOPE_VERSION,
    graph_to_envelope,
    envelope_to_graph,
    GraphEnvelopeRefusal,
    simulate_graph,
)


def _graph() -> dict[str, Any]:
    """Build two populations joined by a seeded random projection."""
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
    restored = envelope_to_graph(graph_to_envelope(_graph()))
    projection = restored["projections"][0]

    assert projection["rule"] == "random"
    assert projection["probability"] == 0.137
    assert projection["seed"] == 7
    assert projection["autapses"] is False


def test_a_population_keeps_the_name_a_reader_gave_it() -> None:
    """Every label was replaced by its identifier, not only names holding LIF."""
    restored = envelope_to_graph(graph_to_envelope(_graph()))

    assert [p["label"] for p in restored["populations"]] == [
        "Excitatory LIF pool",
        "Inhibitory",
    ]


def test_the_export_declares_the_version_that_carries_connectivity() -> None:
    """A reader must be able to tell a faithful document from a lossy one."""
    exported = graph_to_envelope(_graph())

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

    restored = envelope_to_graph(legacy)
    projection = restored["projections"][0]

    assert projection["rule"] == "all_to_all"
    assert "probability" not in projection
    assert [p["label"] for p in restored["populations"]] == ["p1", "p2"]


def test_execution_settings_drives_and_identifiers_survive_json_and_real_simulation() -> None:
    """Preserve the simulation that a saved envelope actually describes."""
    graph = {
        "dt": 0.5,
        "duration": 40.0,
        "seed": 91,
        "populations": [
            {
                "id": "source",
                "type": "population",
                "label": "Driven source",
                "model": "SCLapicqueLIFNeuron",
                "count": 4,
                "neuron_type": "excitatory",
                "params": {"tau": 3.0, "resistance": 2.0, "v_threshold": 0.8},
                "drive": {"kind": "constant", "current": 0.8},
                "position": {"x": 37, "y": 91},
            },
            {
                "id": "target",
                "type": "population",
                "label": "Poisson target",
                "model": "SCLapicqueLIFNeuron",
                "count": 3,
                "neuron_type": "excitatory",
                "params": {"tau": 4.0, "resistance": 2.0},
                "drive": {"kind": "poisson", "rate_hz": 200.0, "weight": 0.4},
                "position": {"x": 301, "y": 118},
            },
        ],
        "projections": [
            {
                "id": "named_random",
                "source": "source",
                "target": "target",
                "rule": "random",
                "probability": 0.6,
                "weight": 0.7,
                "delay": 1.5,
                "autapses": False,
            }
        ],
    }
    saved = deepcopy(graph)
    payload = graph_to_envelope(graph)
    restored = envelope_to_graph(json.loads(json.dumps(payload)))
    assert restored == graph
    before, after = simulate_graph(graph), simulate_graph(restored)
    assert before["success"] and after["success"]
    assert before == after
    assert graph == saved
    # The portable object owns its dictionaries; editing it cannot change the canvas.
    payload["nodes"]["source"]["params"]["tau"] = 99
    restored["populations"][0]["drive"]["current"] = 9
    assert graph == saved
    assert before["n_spikes"] > 0


@pytest.mark.parametrize("version", ["future", 3, None, [], {}])
def test_an_unknown_envelope_version_is_refused(version: Any) -> None:
    """Refuse future or malformed versions before interpreting a Studio envelope."""
    with pytest.raises(GraphEnvelopeRefusal, match="version is unsupported"):
        envelope_to_graph({"format": "sc-neurocore.studio.network-graph", "version": version})


@pytest.mark.parametrize("missing", ["dt", "duration", "seed"])
def test_current_envelope_cannot_silently_replace_missing_execution_settings(missing: str) -> None:
    """Refuse a current envelope whose missing settings would select new defaults."""
    payload = graph_to_envelope(_graph())
    del payload[missing]
    with pytest.raises(GraphEnvelopeRefusal, match="needs dt, duration and seed"):
        envelope_to_graph(payload)
