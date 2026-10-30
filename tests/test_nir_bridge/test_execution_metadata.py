# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public NIR execution contract round trips and refusals

"""Exercise recorded settings and metadata through the real NIR file boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nir
import numpy as np
import pytest

from sc_neurocore.nir_bridge import SCNetwork, from_nir, to_nir
from sc_neurocore.nir_bridge.execution_metadata import (
    EXECUTION_METADATA_KEY,
    EXECUTION_METADATA_SCHEMA,
)


def _graph(metadata: dict[str, object] | None = None) -> Any:
    """Build a driven, nonuniform LIF graph with original attribution metadata."""
    return nir.NIRGraph(
        nodes={
            "drive": nir.Input(input_type=np.array([2])),
            "lif": nir.LIF(
                tau=np.array([2.0, 3.0]),
                r=np.ones(2),
                v_leak=np.array([-0.2, -0.1]),
                v_threshold=np.array([0.7, 0.8]),
                v_reset=np.array([-0.3, -0.4]),
                metadata={"source": "declared-physical-parameters"},
            ),
            "spikes": nir.Output(output_type=np.array([2])),
        },
        edges=[("drive", "lif"), ("lif", "spikes")],
        metadata=metadata if metadata is not None else {"experiment": "round-trip"},
    )


def _contract(**changes: object) -> str:
    """Encode an externally supplied execution contract for public import."""
    return json.dumps(
        {
            "schema_version": EXECUTION_METADATA_SCHEMA,
            "dt": 0.125,
            "reset_mode": "subtract",
            **changes,
        }
    )


def test_a_nir_file_keeps_settings_metadata_and_nontrivial_spike_semantics(
    tmp_path: Path,
) -> None:
    """Default reimport replays the recorded convention without re-entered options."""
    graph = _graph()
    network = from_nir(graph, dt=0.125, reset_mode="subtract")
    path = tmp_path / "network.nir"
    exported = to_nir(network, path)
    restored = SCNetwork.from_nir(path)

    assert restored.dt == 0.125 and restored.reset_mode == "subtract"
    assert exported.metadata["experiment"] == "round-trip"
    assert restored.nir_metadata == exported.metadata
    assert restored.nir_node_metadata["lif"] == {"source": "declared-physical-parameters"}
    assert json.loads(exported.metadata[EXECUTION_METADATA_KEY]) == {
        "schema_version": EXECUTION_METADATA_SCHEMA,
        "dt": 0.125,
        "reset_mode": "subtract",
    }
    assert restored.interchange_notes == []
    assert graph.metadata == {"experiment": "round-trip"}
    expected = [network.step({"drive": np.array([10.0, 12.0])})["spikes"] for _ in range(20)]
    actual = [restored.step({"drive": np.array([10.0, 12.0])})["spikes"] for _ in range(20)]
    np.testing.assert_array_equal(actual, expected)
    assert 0 < np.count_nonzero(expected) < 40


def test_metadata_is_copied_in_both_directions() -> None:
    """Edits to the caller's graph or exported tags cannot rewrite imported custody."""
    graph = _graph()
    network = from_nir(graph)
    graph.metadata["experiment"] = "changed-source"
    graph.nodes["lif"].metadata["source"] = "changed-source"
    exported = to_nir(network)
    exported.metadata["experiment"] = "changed-export"
    exported.nodes["lif"].metadata["source"] = "changed-export"

    assert network.nir_metadata == {"experiment": "round-trip"}
    assert network.nir_node_metadata["lif"] == {"source": "declared-physical-parameters"}


def test_matching_explicit_options_accept_a_recorded_contract() -> None:
    """Explicit options may confirm the graph's execution convention."""
    graph = _graph({EXECUTION_METADATA_KEY: _contract()})
    network = from_nir(graph, dt=0.125, reset_mode="subtract")
    assert network.dt == 0.125 and network.reset_mode == "subtract"


@pytest.mark.parametrize("options", [{"dt": 1.0}, {"reset_mode": "reset"}])
def test_conflicting_import_options_are_refused(options: dict[str, Any]) -> None:
    """An importer cannot silently change a recorded clock or reset convention."""
    with pytest.raises(ValueError, match="conflict with recorded execution"):
        from_nir(_graph({EXECUTION_METADATA_KEY: _contract()}), **options)


@pytest.mark.parametrize(
    "text",
    [
        7,
        "{",
        "null",
        "[]",
        _contract(schema_version="future"),
        _contract(extra="unrecognised"),
        json.dumps({"schema_version": EXECUTION_METADATA_SCHEMA, "dt": 1}),
        _contract(dt=True),
        _contract(dt="0.125"),
        _contract(dt=0),
        _contract(dt=-1),
        _contract(dt=float("inf")),
        _contract(dt=float("nan")),
        _contract(reset_mode=False),
        _contract(reset_mode="delayed"),
        '{"schema_version":"sc-neurocore.nir-execution.v1","dt":0.125,"dt":1,"reset_mode":"subtract"}',
    ],
)
def test_invalid_recorded_settings_refuse_before_execution(text: object) -> None:
    """Malformed or unsupported metadata is refused instead of defaulting silently."""
    with pytest.raises(ValueError, match="NIR (execution|import)"):
        from_nir(_graph({EXECUTION_METADATA_KEY: text}))


def test_a_foreign_graph_keeps_the_documented_defaults() -> None:
    """Graphs without the extension still use timestep one and reset to v_reset."""
    network = from_nir(_graph({}))
    assert network.dt == 1.0 and network.reset_mode == "reset"
    assert network.interchange_notes


def test_non_mapping_graph_metadata_is_refused_before_defaults() -> None:
    """A malformed metadata container cannot silently select a different clock."""
    graph = _graph()
    graph.metadata = None
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        from_nir(graph)


def test_export_refuses_corrupt_existing_execution_metadata() -> None:
    """A modified network cannot publish an unrecognised profile as a valid graph."""
    network = from_nir(_graph())
    network.nir_metadata[EXECUTION_METADATA_KEY] = _contract(schema_version="future")
    with pytest.raises(ValueError, match="unsupported schema"):
        to_nir(network)


def test_nested_graphs_retain_their_metadata_and_share_the_declared_clock() -> None:
    """Profile settings and original attribution survive a nested NIR file graph."""
    inner = _graph({"experiment": "nested"})
    graph = nir.NIRGraph(
        nodes={
            "drive": nir.Input(input_type=np.array([2])),
            "inner": inner,
            "spikes": nir.Output(output_type=np.array([2])),
        },
        edges=[("drive", "inner"), ("inner", "spikes")],
    )
    exported = to_nir(from_nir(graph, dt=0.125, reset_mode="subtract"))
    restored = from_nir(exported)
    assert restored.nodes["inner"].network.dt == 0.125
    assert restored.nodes["inner"].network.reset_mode == "subtract"
    assert restored.nodes["inner"].network.nir_metadata["experiment"] == "nested"


def test_conflicting_nested_execution_clocks_are_refused() -> None:
    """The single-clock executor cannot silently override a nested recorded clock."""
    inner = _graph({EXECUTION_METADATA_KEY: _contract()})
    graph = nir.NIRGraph(
        nodes={
            "drive": nir.Input(input_type=np.array([2])),
            "inner": inner,
            "spikes": nir.Output(output_type=np.array([2])),
        },
        edges=[("drive", "inner"), ("inner", "spikes")],
    )
    with pytest.raises(ValueError, match="conflict with recorded execution"):
        from_nir(graph)
