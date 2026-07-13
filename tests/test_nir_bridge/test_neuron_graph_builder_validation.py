# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware neuron graph builder validation

"""Exercise builder fallbacks and malformed graph rejection through its public API."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_scnetwork
from sc_neurocore.nir_bridge.node_map import (
    SCAffineNode,
    SCCubaLIFNode,
    SCCubaLINode,
    SCFlattenNode,
    SCIFNode,
    SCInputNode,
    SCLIFNode,
    SCOutputNode,
)
from sc_neurocore.nir_bridge.parser import SCNetwork, _UnitDelayNode


def _lif(name: str = "lif", width: int = 2) -> SCLIFNode:
    """Build a deterministic parsed LIF node."""
    return SCLIFNode(
        name,
        width,
        tau=np.full(width, 20.0),
        r=np.ones(width),
        v_leak=np.zeros(width),
        v_threshold=np.ones(width),
        v_reset=np.zeros(width),
        dt=0.25,
    )


def _affine(name: str = "affine", rows: int = 2, columns: int = 2) -> SCAffineNode:
    """Build a deterministic parsed affine node."""
    return SCAffineNode(name, np.ones((rows, columns), dtype=np.float32), np.zeros(rows))


@pytest.mark.parametrize(
    ("node", "neuron_type", "parameter_keys"),
    [
        (
            SCIFNode(
                "if",
                1,
                r=np.ones(1),
                v_threshold=np.ones(1),
                v_reset=np.zeros(1),
            ),
            "if",
            {"r", "v_threshold", "v_reset"},
        ),
        (
            SCCubaLIFNode(
                "cuba_lif",
                1,
                tau_syn=np.ones(1),
                tau_mem=np.ones(1),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
                v_reset=np.zeros(1),
                w_in=np.ones(1),
            ),
            "cuba_lif",
            {"r", "v_leak", "v_threshold", "v_reset", "tau_syn", "tau_mem", "w_in"},
        ),
        (
            SCCubaLINode(
                "cuba_li",
                1,
                tau_syn=np.ones(1),
                tau_mem=np.ones(1),
                r=np.ones(1),
                v_leak=np.zeros(1),
                w_in=np.ones(1),
            ),
            "cuba_li",
            {"r", "v_leak", "tau_syn", "tau_mem", "w_in"},
        ),
    ],
)
def test_unconnected_neuron_populations_keep_type_specific_parameters(
    node: Any,
    neuron_type: str,
    parameter_keys: set[str],
) -> None:
    """Build boundary-free populations and retain their canonical parameters."""
    graph = from_scnetwork(SCNetwork(nodes={node.name: node}), dt=None)
    population = graph.populations[0]

    assert graph.input_pop == node.name
    assert graph.output_pop == node.name
    assert graph.connections == []
    assert population.neuron_type == neuron_type
    assert set(population.params) == parameter_keys


def test_weight_without_neuron_destination_is_ignored_with_unsupported_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ignore a terminal affine node and report an unsupported parsed node."""

    class UnknownParsedNode:
        pass

    nodes = {
        "input": SCInputNode("input", (2,)),
        "affine": _affine(),
        "output": SCOutputNode("output", (2,)),
        "lif": _lif(),
        "unknown": UnknownParsedNode(),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[("input", "affine"), ("affine", "output")],
        input_nodes=["input"],
        output_nodes=["output"],
    )
    with caplog.at_level(logging.WARNING, logger="sc_neurocore.nir_bridge.neuron_graph"):
        graph = from_scnetwork(network)

    assert graph.connections == []
    assert "Skipping unsupported node type UnknownParsedNode" in caplog.text


def test_unsupported_weight_predecessor_uses_exact_predecessor_name() -> None:
    """Retain a named external source when it is not a recognised population."""

    class ExternalSource:
        pass

    nodes = {"external": ExternalSource(), "affine": _affine(), "lif": _lif()}
    graph = from_scnetwork(
        SCNetwork(nodes=nodes, edges=[("external", "affine"), ("affine", "lif")])
    )
    assert graph.connections[0].src == "external"


def test_weight_without_predecessor_uses_default_input_name() -> None:
    """Use the documented synthetic input name for an unconnected affine node."""
    graph = from_scnetwork(
        SCNetwork(nodes={"affine": _affine(), "lif": _lif()}, edges=[("affine", "lif")])
    )
    assert graph.connections[0].src == "input"


def test_post_weight_flatten_must_match_weight_rows() -> None:
    """Reject a shape-preserving flatten wider than the affine output."""
    nodes = {
        "input": SCInputNode("input", (2,)),
        "affine": _affine(rows=2),
        "flatten": SCFlattenNode("flatten", 0, -1, (3,), (3,)),
        "lif": _lif(width=3),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[("input", "affine"), ("affine", "flatten"), ("flatten", "lif")],
    )
    with pytest.raises(ValueError, match="Flatten input width 3.*weight output width 2"):
        from_scnetwork(network)


def test_recurrent_metadata_without_weight_source_is_ignored() -> None:
    """Ignore a stale recurrent-map entry that names no weight node."""
    lif = _lif()
    network = SCNetwork(nodes={"lif": lif}, _recurrent_map={"delay": "missing"})
    assert from_scnetwork(network).connections == []


def test_recurrent_weight_without_neuron_predecessor_is_ignored() -> None:
    """Ignore recurrent metadata whose weight source has no neuron origin."""
    nodes = {
        "input": SCInputNode("input", (2,)),
        "affine": _affine(),
        "delay": _UnitDelayNode("delay"),
        "lif": _lif(),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[("input", "affine"), ("delay", "lif")],
        _recurrent_map={"delay": "affine"},
    )
    assert from_scnetwork(network).connections == []


def test_recurrent_weight_without_neuron_destination_is_ignored() -> None:
    """Ignore recurrent metadata whose delay reaches no neuron destination."""
    nodes = {
        "lif": _lif(),
        "affine": _affine(),
        "delay": _UnitDelayNode("delay"),
        "output": SCOutputNode("output", (2,)),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[("lif", "affine"), ("delay", "output")],
        _recurrent_map={"delay": "affine"},
    )
    assert from_scnetwork(network).connections == []
