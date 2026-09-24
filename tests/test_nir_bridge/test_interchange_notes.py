# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR interchange note tests

"""Every assumption, approximation and omission at the NIR boundary is named."""

from __future__ import annotations

from pathlib import Path

import h5py
import nir
import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_nir, to_nir
from sc_neurocore.nir_bridge.interchange_notes import (
    InterchangeNote,
    export_notes,
    read_nir_file_version,
)
from sc_neurocore.studio.nir_compile import compile_nir_file_bytes


def _lif() -> object:
    return nir.LIF(
        tau=np.array([10.0]),
        r=np.array([1.0]),
        v_leak=np.array([0.0]),
        v_threshold=np.array([1.0]),
        v_reset=np.array([0.0]),
    )


def _io(shape: int = 1) -> tuple[object, object]:
    return (
        nir.Input(input_type={"input": np.array([shape])}),
        nir.Output(output_type={"output": np.array([shape])}),
    )


def _spiking_graph(**graph_options: object) -> object:
    source, sink = _io()
    return nir.NIRGraph(
        nodes={"in": source, "lif": _lif(), "out": sink},
        edges=[("in", "lif"), ("lif", "out")],
        **graph_options,
    )


def _kinds(notes: list[InterchangeNote]) -> list[tuple[str, str, str]]:
    return [(note.kind, note.subject, note.detail) for note in notes]


def test_every_import_names_the_timestep_and_the_reset_rule_it_supplied() -> None:
    """NIR stores neither, so both are recorded as assumptions."""
    network = from_nir(_spiking_graph(), dt=0.5, reset_mode="subtract")

    assert network.dt == 0.5 and network.reset_mode == "subtract"
    assert network.nir_version is None
    assert _kinds(network.interchange_notes) == [
        ("assumed", "", "NIR stores no timestep; dynamics use dt=0.5 supplied to the importer"),
        (
            "assumed",
            "",
            "NIR stores no reset rule; spiking nodes use 'subtract' "
            "(the NIR convention is 'reset' to v_reset)",
        ),
    ]


def test_a_graph_without_spiking_nodes_assumes_no_reset_rule() -> None:
    """A purely linear graph has no reset to assume."""
    source, sink = _io()
    graph = nir.NIRGraph(
        nodes={"in": source, "scale": nir.Scale(scale=np.array([2.0])), "out": sink},
        edges=[("in", "scale"), ("scale", "out")],
    )
    notes = from_nir(graph).interchange_notes

    assert [note.detail.split(";")[0] for note in notes] == ["NIR stores no timestep"]


def test_an_unknown_reset_mode_is_refused() -> None:
    """A reset rule the nodes do not implement is not silently replaced."""
    with pytest.raises(ValueError, match="reset_mode must be one of"):
        from_nir(_spiking_graph(), reset_mode="clamp")


def test_metadata_extra_ports_fractional_delays_and_recurrence_are_named() -> None:
    """Each construct the bridge cannot carry or realise exactly gets its own note."""
    graph = nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type={"input": np.array([1]), "extra": np.array([1])}),
            "delay": nir.Delay(delay=np.array([2.5])),
            "whole": nir.Delay(delay=np.array([3.0])),
            "lif": _lif(),
            "out": nir.Output(output_type={"output": np.array([1]), "extra": np.array([1])}),
        },
        edges=[
            ("in", "delay"),
            ("delay", "whole"),
            ("whole", "lif"),
            ("lif", "lif"),
            ("lif", "out"),
        ],
        metadata={"source": "snnTorch", "author": "x"},
        type_check=False,
    )
    notes = _kinds(from_nir(graph, dt=1.0).interchange_notes)

    assert ("not-carried", "", "graph metadata is not imported: author, source") in notes
    assert ("approximated", "in", "only the first input port's shape is used") in notes
    assert ("approximated", "out", "only the first output port's shape is used") in notes
    assert (
        "approximated",
        "delay",
        "delay of [2.5] timesteps realised as [2] whole timesteps",
    ) in notes
    assert not any(subject == "whole" for _kind, subject, _detail in notes)
    assert ("assumed", "lif->lif", "recurrent edge realised with a one-timestep delay") in notes


def test_notes_inside_subgraphs_carry_the_subgraph_path() -> None:
    """A nested graph's notes name the node inside the subgraph, and multi-port use."""
    inner_single = nir.NIRGraph(
        nodes={
            "i": nir.Input(input_type={"input": np.array([1])}),
            "d": nir.Delay(delay=np.array([0.4])),
            "o": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("i", "d"), ("d", "o")],
        metadata={"note": "inner"},
    )
    inner_multi = nir.NIRGraph(
        nodes={
            "a": nir.Input(input_type={"input": np.array([1])}),
            "b": nir.Input(input_type={"input": np.array([1])}),
            "oa": nir.Output(output_type={"output": np.array([1])}),
            "ob": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("a", "oa"), ("b", "ob")],
        type_check=False,
    )
    source, sink = _io()
    graph = nir.NIRGraph(
        nodes={"in": source, "single": inner_single, "multi": inner_multi, "out": sink},
        edges=[("in", "single"), ("single", "multi"), ("multi", "out")],
        type_check=False,
    )
    notes = _kinds(from_nir(graph).interchange_notes)

    assert ("not-carried", "single", "graph metadata is not imported: note") in notes
    assert (
        "approximated",
        "single.d",
        "delay of [0.4] timesteps realised as [0] whole timesteps",
    ) in notes
    assert (
        "approximated",
        "multi",
        "multi-port subgraph used as one node: only its first input and "
        "first output port are connected in the enclosing graph",
    ) in notes


def test_a_nir_file_records_its_version(tmp_path: Path) -> None:
    """The version a written file stores is read back and kept on the network."""
    path = tmp_path / "graph.nir"
    nir.write(str(path), _spiking_graph())

    network = from_nir(path)

    assert network.nir_version == read_nir_file_version(path)
    assert network.nir_version == nir.version


def test_a_file_without_a_version_reports_none(tmp_path: Path) -> None:
    """An HDF5 document with no version dataset has no recorded version."""
    path = tmp_path / "bare.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("node")

    assert read_nir_file_version(path) is None


def test_exporting_names_what_nir_cannot_store() -> None:
    """Timestep, subtract reset, node state and pooling input shape are not written."""
    source, sink = _io(4)
    pooled = nir.NIRGraph(
        nodes={
            "i": nir.Input(input_type={"input": np.array([1, 4, 4])}),
            "pool": nir.SumPool2d(
                kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0])
            ),
            "o": nir.Output(output_type={"output": np.array([1, 2, 2])}),
        },
        edges=[("i", "pool"), ("pool", "o")],
        type_check=False,
    )
    graph = nir.NIRGraph(
        nodes={"in": source, "lif": _lif(), "sub": pooled, "out": sink},
        edges=[("in", "lif"), ("lif", "sub"), ("sub", "out")],
        type_check=False,
    )
    network = from_nir(graph, dt=0.25, reset_mode="subtract")

    assert _kinds(export_notes(network)) == [
        (
            "not-carried",
            "",
            "NIR stores no timestep; dt=0.25 must be supplied again on import",
        ),
        (
            "not-carried",
            "",
            "NIR stores no reset rule; the subtract reset is lost and importers "
            "apply the NIR convention of reset to v_reset",
        ),
        (
            "not-carried",
            "",
            "NIR stores no node state; membrane, current and delay-buffer state is "
            "not written and an importer starts from the initial state",
        ),
        ("not-carried", "sub.pool", "pooling input shape (input_type) is not written"),
    ]


def test_a_stateless_reset_graph_exports_with_only_the_timestep_note() -> None:
    """With nothing else lost, only the timestep remains to be supplied."""
    source, sink = _io()
    graph = nir.NIRGraph(
        nodes={"in": source, "scale": nir.Scale(scale=np.array([2.0])), "out": sink},
        edges=[("in", "scale"), ("scale", "out")],
    )
    network = from_nir(graph)

    assert [note.kind for note in export_notes(network)] == ["not-carried"]
    assert isinstance(to_nir(network), nir.NIRGraph)


def test_notes_serialise_for_the_studio_and_the_compile_result_carries_them(
    tmp_path: Path,
) -> None:
    """The Studio compile result reports the version and the import notes."""
    path = tmp_path / "graph.nir"
    nir.write(str(path), _spiking_graph())

    result = compile_nir_file_bytes(path.read_bytes())

    assert result["nir_version"] == nir.version
    assert result["interchange_notes"][0] == {
        "kind": "assumed",
        "subject": "",
        "detail": "NIR stores no timestep; dynamics use dt=1.0 supplied to the importer",
    }


def test_a_spiking_node_inside_a_subgraph_still_needs_the_reset_assumption() -> None:
    """The reset rule is assumed wherever a spiking node sits in the hierarchy."""
    source, sink = _io()
    graph = nir.NIRGraph(
        nodes={"in": source, "inner": _spiking_graph(), "out": sink},
        edges=[("in", "inner"), ("inner", "out")],
        type_check=False,
    )
    details = [note.detail for note in from_nir(graph).interchange_notes]

    assert any(detail.startswith("NIR stores no reset rule") for detail in details)
