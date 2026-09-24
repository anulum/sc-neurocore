# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the Studio network graph as a real NIR file

"""Every test here writes or reads a real NIR (HDF5) file through ``nir``."""

from __future__ import annotations

import base64
import copy
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir
from sc_neurocore.studio.network_execution import connectivity_arrays
from sc_neurocore.studio.network_graph import simulate_graph
from sc_neurocore.studio.network_graph_spec import GraphRejected, resolve_graph
from sc_neurocore.studio.network_nir import (
    NIR_EXPORT_FILENAME,
    STUDIO_METADATA_KEY,
    STUDIO_NIR_EXPORT_SCHEMA_VERSION,
    NIRMappingRefused,
    graph_to_nir_file,
    nir_file_to_graph,
)

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _population(pop_id: str, model: str, count: int, **fields: Any) -> dict[str, Any]:
    population: dict[str, Any] = {
        "id": pop_id,
        "type": "population",
        "label": pop_id.upper(),
        "model": model,
        "count": count,
        "neuron_type": "excitatory",
        "params": {},
        "drive": {"kind": "none"},
        "position": {"x": 0, "y": 0},
    }
    population.update(fields)
    return population


def _network() -> dict[str, Any]:
    """A driven LIF population projecting at random onto an IF population."""
    return {
        "populations": [
            _population(
                "lif",
                "SCLapicqueLIFNeuron",
                4,
                params={"tau": 12.0, "resistance": 2.0, "v_threshold": 0.8},
                drive={"kind": "constant", "current": 0.6},
                position={"x": 10, "y": 20},
            ),
            _population(
                "acc",
                "PerfectIntegratorNeuron",
                3,
                params={"c_m": 2.0, "v_threshold": 1.5},
                position={"x": 210, "y": 20},
            ),
        ],
        "projections": [
            {
                "id": "p_ff",
                "source": "lif",
                "target": "acc",
                "weight": 0.4,
                "delay": 2.0,
                "rule": "random",
                "probability": 0.6,
                "seed": 3,
            }
        ],
        "dt": 1.0,
        "duration": 30.0,
        "seed": 9,
    }


def _encode(graph: Any) -> str:
    return base64.b64encode(_write(graph)).decode("ascii")


def _write(graph: Any) -> bytes:
    """Write an in-memory NIR graph to bytes, as another tool would."""
    buffer = io.BytesIO()
    nir.write(buffer, graph)
    return buffer.getvalue()


def _read(export_bytes: bytes, tmp_path: Path) -> Any:
    path = tmp_path / NIR_EXPORT_FILENAME
    path.write_bytes(export_bytes)
    return nir.read(path)


def _lif(count: int, **overrides: Any) -> Any:
    fields = {
        "tau": np.full(count, 10.0),
        "r": np.full(count, 1.5),
        "v_leak": np.full(count, 0.1),
        "v_threshold": np.full(count, 1.0),
        "v_reset": np.full(count, 0.0),
    }
    fields.update(overrides)
    return nir.LIF(**fields)


def _if(count: int, **overrides: Any) -> Any:
    fields = {"r": np.full(count, 0.5), "v_threshold": np.full(count, 2.0)}
    fields.update(overrides)
    return nir.IF(**fields)


def _output(count: int) -> Any:
    return nir.Output(output_type={"output": np.array([count])})


def _foreign(nodes: dict[str, Any], edges: list[tuple[str, str]]) -> str:
    return _encode(nir.NIRGraph(nodes=nodes, edges=edges, type_check=False))


class TestExport:
    def test_the_export_is_a_real_nir_file_of_the_network(self, tmp_path):
        exported = graph_to_nir_file(_network())
        assert exported.content.startswith(HDF5_SIGNATURE)
        assert exported.nir_version == nir.version

        read = _read(exported.content, tmp_path)
        lif, acc = read.nodes["lif"], read.nodes["acc"]
        assert isinstance(lif, nir.LIF)
        np.testing.assert_array_equal(lif.tau, np.full(4, 12.0))
        np.testing.assert_array_equal(lif.r, np.full(4, 2.0))
        np.testing.assert_array_equal(lif.v_leak, np.zeros(4))
        np.testing.assert_array_equal(lif.v_threshold, np.full(4, 0.8))
        assert isinstance(acc, nir.IF)
        np.testing.assert_array_equal(acc.r, np.full(3, 0.5))
        np.testing.assert_array_equal(acc.v_threshold, np.full(3, 1.5))
        assert isinstance(read.nodes["input_lif"], nir.Input)
        assert "input_acc" not in read.nodes
        assert isinstance(read.nodes["output_acc"], nir.Output)
        # The runtime delivers a spike delay + 1 steps after it was emitted.
        np.testing.assert_array_equal(read.nodes["p_ff_delay"].delay, np.full(3, 3.0))
        assert sorted(read.edges) == sorted(
            [
                ("input_lif", "lif"),
                ("lif", "output_lif"),
                ("acc", "output_acc"),
                ("lif", "p_ff_weight"),
                ("p_ff_weight", "p_ff_delay"),
                ("p_ff_delay", "acc"),
            ]
        )
        graph_metadata = json.loads(read.metadata[STUDIO_METADATA_KEY])
        assert graph_metadata == {
            "schema_version": STUDIO_NIR_EXPORT_SCHEMA_VERSION,
            "dt": 1.0,
            "duration": 30.0,
            "seed": 9,
            "time_unit": "ms",
        }

    def test_the_weight_is_the_connectivity_the_runtime_realises(self, tmp_path):
        graph = _network()
        weight = _read(graph_to_nir_file(graph).content, tmp_path).nodes["p_ff_weight"].weight
        projection = resolve_graph(graph).projections[0]
        indptr, indices, data, _removed = connectivity_arrays(projection, 4, 3)

        assert weight.shape == (3, 4)
        realised = {
            (int(target), source)
            for source in range(4)
            for target in indices[indptr[source] : indptr[source + 1]]
        }
        assert {tuple(map(int, pair)) for pair in zip(*np.nonzero(weight))} == {
            (target, source) for target, source in realised
        }
        assert set(np.unique(weight[weight != 0])) == set(np.unique(data))

    def test_the_notes_state_what_the_file_does_not_carry(self):
        notes = graph_to_nir_file(_network()).notes
        assert any("milliseconds" in note for note in notes)
        assert any(
            note.startswith("population lif:")
            and "v >= v_threshold, NIR LIF at v > v_threshold" in note
            for note in notes
        )
        assert any(
            "population acc: profile sc_inclusive fires at v >= v_threshold" in note
            for note in notes
        )
        assert any("constant current 0.6" in note and "input_lif" in note for note in notes)
        assert any("1-step spike propagation latency at dt = 1.0 ms" in note for note in notes)
        assert not any("initial membrane" in note for note in notes)

    def test_a_poisson_drive_and_initial_state_are_named_in_the_notes(self):
        graph = _network()
        graph["populations"][0]["params"]["v"] = 0.3
        graph["populations"][1]["params"]["v"] = 0.2
        graph["populations"][1]["drive"] = {
            "kind": "poisson",
            "rate_hz": 40.0,
            "weight": 0.5,
            "seed": 4,
        }
        notes = graph_to_nir_file(graph).notes
        assert any("population lif: initial membrane 0.3 is not carried" in n for n in notes)
        assert any("population acc: initial membrane 0.2 is not carried" in n for n in notes)
        assert any("Poisson input at 40.0 Hz, weight 0.5, seed 4" in n for n in notes)

    def test_the_public_dict_carries_the_file_bytes(self):
        exported = graph_to_nir_file(_network())
        public = exported.to_public_dict()
        assert public["schema_version"] == STUDIO_NIR_EXPORT_SCHEMA_VERSION
        assert public["filename"] == NIR_EXPORT_FILENAME
        assert public["media_type"] == "application/x-hdf5"
        assert public["nir_version"] == nir.version
        assert base64.b64decode(public["content_base64"]) == exported.content
        assert public["notes"] == list(exported.notes)

    def test_a_model_without_an_nir_primitive_is_refused_by_name(self):
        graph = _network()
        graph["populations"][1]["model"] = "AdExNeuron"
        graph["populations"][1]["params"] = {}
        with pytest.raises(NIRMappingRefused, match=r"population acc: model AdExNeuron has no NIR"):
            graph_to_nir_file(graph)

    def test_a_refused_model_with_a_profile_names_the_profile(self):
        graph = _network()
        graph["populations"][1]["model"] = "QuadraticIFNeuron"
        graph["populations"][1]["params"] = {}
        with pytest.raises(NIRMappingRefused, match=r"QuadraticIFNeuron \(profile sc_symmetric\)"):
            graph_to_nir_file(graph)

    def test_an_invalid_graph_is_rejected_before_export(self):
        with pytest.raises(GraphRejected):
            graph_to_nir_file([])

    def test_a_population_without_params_drive_or_position_exports(self, tmp_path):
        graph = {
            "populations": [
                {
                    "id": "bare",
                    "type": "population",
                    "label": "Bare",
                    "model": "PerfectIntegratorNeuron",
                    "count": 2,
                    "neuron_type": "excitatory",
                }
            ],
            "projections": [],
            "dt": 1.0,
        }
        read = _read(graph_to_nir_file(graph).content, tmp_path)
        metadata = json.loads(read.nodes["bare"].metadata[STUDIO_METADATA_KEY])
        assert metadata["params"] == {}
        assert metadata["drive"] == {"kind": "none"}
        assert metadata["position"] is None

    def test_without_the_nir_package_export_says_which_extra_to_install(self, monkeypatch):
        """``sys.modules[name] = None`` is Python's own import block: the extra is absent."""
        monkeypatch.setitem(sys.modules, "nir", None)
        with pytest.raises(NIRMappingRefused, match=r"install sc-neurocore\[nir\]"):
            graph_to_nir_file(_network())


class TestStudioRoundTrip:
    def test_a_studio_file_reads_back_as_the_same_network(self):
        graph = _network()
        imported = nir_file_to_graph(_export_base64(graph))
        assert imported["origin"] == "studio"
        assert imported["notes"] == []
        expected = copy.deepcopy(graph)
        # The runtime's resolved autapse decision is written out explicitly.
        expected["projections"][0]["autapses"] = False
        assert imported["graph"] == expected

    def test_the_graph_order_and_the_seeds_derived_from_it_survive(self):
        """HDF5 lists nodes by name; the order Studio seeds from must come back."""
        graph = _network()
        graph["populations"].reverse()
        graph["populations"][0]["id"] = "zeta"
        graph["projections"][0]["target"] = "zeta"
        del graph["projections"][0]["seed"]
        graph["populations"][0]["drive"] = {"kind": "poisson", "rate_hz": 30.0, "weight": 0.2}
        graph["populations"][1]["drive"] = {"kind": "poisson", "rate_hz": 30.0, "weight": 0.2}

        imported = nir_file_to_graph(_export_base64(graph))["graph"]
        assert [p["id"] for p in imported["populations"]] == ["zeta", "lif"]
        original, rebuilt = resolve_graph(graph), resolve_graph(imported)
        assert [p.drive.seed for p in rebuilt.populations] == [
            p.drive.seed for p in original.populations
        ]
        assert rebuilt.projections[0].seed == original.projections[0].seed
        assert simulate_graph(imported)["populations"] == simulate_graph(graph)["populations"]

    def test_an_all_to_all_poisson_network_reads_back(self):
        graph = _network()
        graph["populations"][1]["drive"] = {
            "kind": "poisson",
            "rate_hz": 25.0,
            "weight": 0.3,
            "seed": 8,
        }
        projection = graph["projections"][0]
        del projection["probability"]
        projection.update(rule="all_to_all", delay=0.0)
        imported = nir_file_to_graph(_export_base64(graph))["graph"]
        assert imported["populations"][1]["drive"] == graph["populations"][1]["drive"]
        assert imported["projections"][0]["rule"] == "all_to_all"
        assert "probability" not in imported["projections"][0]
        assert resolve_graph(imported) == resolve_graph(graph)

    def test_a_population_without_a_position_is_laid_out(self):
        graph = _network()
        for population in graph["populations"]:
            del population["position"]
        populations = nir_file_to_graph(_export_base64(graph))["graph"]["populations"]
        assert [p["position"] for p in populations] == [{"x": 0, "y": 0}, {"x": 200, "y": 0}]

    def test_an_edited_tensor_is_refused(self, tmp_path):
        read = _read(graph_to_nir_file(_network()).content, tmp_path)
        read.nodes["lif"].tau = np.full(4, 13.0)
        with pytest.raises(NIRMappingRefused, match="node lif does not match"):
            nir_file_to_graph(base64.b64encode(_write(read)).decode("ascii"))

    def test_a_resized_tensor_is_refused(self, tmp_path):
        """NIR's own type check on reading refuses it before Studio compares."""
        read = _read(graph_to_nir_file(_network()).content, tmp_path)
        read.nodes["p_ff_delay"] = nir.Delay(
            delay=np.full(5, 3.0), metadata=read.nodes["p_ff_delay"].metadata
        )
        with pytest.raises(NIRMappingRefused, match="type mismatch"):
            nir_file_to_graph(base64.b64encode(_write(read)).decode("ascii"))

    def test_a_replaced_primitive_is_refused(self, tmp_path):
        read = _read(graph_to_nir_file(_network()).content, tmp_path)
        metadata = read.nodes["acc"].metadata
        read.nodes["acc"] = _lif(3)
        read.nodes["acc"].metadata = metadata
        with pytest.raises(NIRMappingRefused, match="node acc does not match"):
            nir_file_to_graph(base64.b64encode(_write(read)).decode("ascii"))

    def test_a_removed_edge_is_refused(self, tmp_path):
        read = _read(graph_to_nir_file(_network()).content, tmp_path)
        read.edges.remove(("lif", "output_lif"))
        with pytest.raises(NIRMappingRefused, match="nodes or edges differ"):
            nir_file_to_graph(base64.b64encode(_write(read)).decode("ascii"))

    @pytest.mark.parametrize(
        ("graph_metadata", "reason"),
        [
            (
                json.dumps({"schema_version": "sc-neurocore.studio.nir-export.v0"}),
                "unknown version",
            ),
            ("{not json", "the graph: its Studio metadata is not JSON"),
            (json.dumps([1, 2]), "the graph: its Studio metadata is not an object"),
            (
                json.dumps({"schema_version": STUDIO_NIR_EXPORT_SCHEMA_VERSION, "dt": 1.0}),
                "lacks the field 'duration'",
            ),
        ],
    )
    def test_unreadable_graph_metadata_is_refused(self, tmp_path, graph_metadata, reason):
        read = _read(graph_to_nir_file(_network()).content, tmp_path)
        read.metadata[STUDIO_METADATA_KEY] = graph_metadata
        with pytest.raises(NIRMappingRefused, match=reason):
            nir_file_to_graph(base64.b64encode(_write(read)).decode("ascii"))

    def test_graph_metadata_written_as_a_group_is_read(self, tmp_path):
        """HDF5 stores a dict as a group; the version check still applies to it."""
        read = _read(graph_to_nir_file(_network()).content, tmp_path)
        read.metadata[STUDIO_METADATA_KEY] = {"schema_version": "other"}
        with pytest.raises(NIRMappingRefused, match="unknown version"):
            nir_file_to_graph(base64.b64encode(_write(read)).decode("ascii"))

    @pytest.mark.parametrize(
        ("node", "entry", "reason"),
        [
            ("acc", {"kind": "synapse"}, "node acc: its Studio metadata names an unknown kind"),
            ("acc", {"kind": "population"}, "lacks the field 'index'"),
            ("acc", {"kind": "population", "index": 1}, "lacks the field 'label'"),
            (
                "p_ff_weight",
                {"kind": "projection", "index": 0, "id": "p_ff"},
                "lacks the field 'source'",
            ),
        ],
    )
    def test_unreadable_node_metadata_is_refused(self, tmp_path, node, entry, reason):
        read = _read(graph_to_nir_file(_network()).content, tmp_path)
        read.nodes[node].metadata[STUDIO_METADATA_KEY] = json.dumps(entry)
        with pytest.raises(NIRMappingRefused, match=reason):
            nir_file_to_graph(base64.b64encode(_write(read)).decode("ascii"))

    def test_metadata_describing_an_invalid_network_is_refused(self, tmp_path):
        read = _read(graph_to_nir_file(_network()).content, tmp_path)
        entry = json.loads(read.nodes["acc"].metadata[STUDIO_METADATA_KEY])
        entry["count"] = 0
        read.nodes["acc"].metadata[STUDIO_METADATA_KEY] = json.dumps(entry)
        with pytest.raises(NIRMappingRefused, match="does not validate"):
            nir_file_to_graph(base64.b64encode(_write(read)).decode("ascii"))

    def test_the_studio_network_runs_the_same_through_the_nir_bridge(self, tmp_path):
        """The exported file, executed by the NIR bridge, spikes as Studio does."""
        graph = {
            "populations": [
                _population(
                    "src",
                    "PerfectIntegratorNeuron",
                    4,
                    params={"c_m": 1.0, "v_threshold": 1.0},
                    drive={"kind": "constant", "current": 0.37},
                ),
                _population(
                    "dst",
                    "PerfectIntegratorNeuron",
                    3,
                    params={"c_m": 1.0, "v_threshold": 1.0},
                ),
            ],
            "projections": [
                {
                    "id": "p",
                    "source": "src",
                    "target": "dst",
                    "weight": 0.53,
                    "delay": 2.0,
                    "rule": "random",
                    "probability": 0.7,
                    "seed": 11,
                }
            ],
            "dt": 1.0,
            "duration": 40.0,
            "seed": 5,
        }
        result = simulate_graph(graph)
        steps = result["n_steps"]
        path = tmp_path / NIR_EXPORT_FILENAME
        path.write_bytes(graph_to_nir_file(graph).content)
        outputs = from_nir(str(path), dt=1.0).run({"input_src": np.full(4, 0.37)}, steps=steps)

        for population in result["populations"]:
            studio = np.zeros((steps, population["count"]), dtype=int)
            studio[population["events"]["step"], population["events"]["neuron"]] = 1
            bridged = np.asarray(outputs[f"output_{population['id']}"]).astype(int)
            assert studio.any(), population["id"]
            np.testing.assert_array_equal(bridged, studio)


def _export_base64(graph: dict[str, Any]) -> str:
    return str(graph_to_nir_file(copy.deepcopy(graph)).to_public_dict()["content_base64"])


class TestForeignImport:
    def test_lif_and_if_populations_joined_by_linear_weights_are_read(self):
        content = _foreign(
            {
                "in": nir.Input(input_type={"input": np.array([2])}),
                "a": _lif(2),
                "b": _if(3, v_reset=np.full(3, 0.25)),
                "w_ab": nir.Linear(weight=np.full((3, 2), 0.7)),
                "d_ab": nir.Delay(delay=np.full(3, 4.0)),
                "w_ba": nir.Linear(weight=np.full((2, 3), -0.2)),
                "out": _output(3),
            },
            [
                ("in", "a"),
                ("a", "w_ab"),
                ("w_ab", "d_ab"),
                ("d_ab", "b"),
                ("b", "w_ba"),
                ("w_ba", "a"),
                ("b", "out"),
            ],
        )
        imported = nir_file_to_graph(content)
        assert imported["origin"] == "foreign"
        graph = imported["graph"]
        a, b = graph["populations"]
        assert (a["model"], a["count"]) == ("SCLapicqueLIFNeuron", 2)
        assert a["params"] == {
            "tau": 10.0,
            "resistance": 1.5,
            "v_rest": 0.1,
            "v_threshold": 1.0,
            "v_reset": 0.0,
        }
        assert a["neuron_type"] == "excitatory"
        assert (b["model"], b["count"], b["neuron_type"]) == (
            "PerfectIntegratorNeuron",
            3,
            "inhibitory",
        )
        assert b["params"] == {"c_m": 2.0, "v_threshold": 2.0, "v_reset": 0.25}
        assert graph["projections"] == [
            {
                "id": "w_ab",
                "source": "a",
                "target": "b",
                "weight": 0.7,
                "delay": 4.0 - 0.1,
                "rule": "all_to_all",
            },
            {
                "id": "w_ba",
                "source": "b",
                "target": "a",
                "weight": -0.2,
                "delay": 0.0,
                "rule": "all_to_all",
            },
        ]
        notes = imported["notes"]
        assert any("milliseconds" in note for note in notes)
        assert (
            "a: NIR LIF fires at v > v_threshold, SCLapicqueLIFNeuron at v >= v_threshold" in notes
        )
        assert (
            "b: NIR IF fires at v > v_threshold, PerfectIntegratorNeuron at v >= v_threshold"
            in notes
        )
        assert any(note.startswith("w_ba: the Studio runtime adds a 1-step") for note in notes)
        assert "b: read as inhibitory, its outgoing weights are negative" in notes
        assert not any(note.startswith("w_ab:") for note in notes)
        resolve_graph(graph)

    @pytest.mark.parametrize(
        ("nodes", "edges", "reason"),
        [
            (
                {"a": _lif(2, tau=np.array([10.0, 11.0]))},
                [],
                "a: tau differs between neurons",
            ),
            (
                {"a": nir.IF(r=np.array([]), v_threshold=np.array([]))},
                [],
                "a: r differs between neurons",
            ),
            (
                {
                    "a": nir.CubaLIF(
                        tau_syn=np.ones(2),
                        tau_mem=np.ones(2),
                        r=np.ones(2),
                        v_leak=np.zeros(2),
                        v_threshold=np.ones(2),
                    )
                },
                [],
                "a: CubaLIF has no Studio population or projection",
            ),
            (
                {"a": _if(2), "b": _if(2), "d": nir.Delay(delay=np.full(2, 2.0))},
                [("a", "d"), ("d", "b")],
                "a -> d: a population may feed only nir.Linear or nir.Output",
            ),
            (
                {
                    "a": _if(2),
                    "b": _if(2),
                    "w": nir.Linear(weight=np.ones((2, 2))),
                    "d": nir.Delay(delay=np.full(2, 0.05)),
                },
                [("a", "w"), ("w", "d"), ("d", "b")],
                "d: a delay of 0.05 ms is shorter than the Studio runtime's 0.1 ms",
            ),
            (
                {
                    "a": _if(2),
                    "b": _if(2),
                    "w_up": nir.Linear(weight=np.ones((2, 2))),
                    "w_down": nir.Linear(weight=np.full((2, 2), -1.0)),
                },
                [("a", "w_up"), ("w_up", "b"), ("a", "w_down"), ("w_down", "b")],
                "a: its outgoing weights have both signs",
            ),
            (
                {
                    "a": _if(2),
                    "b": _if(2),
                    "c": _if(2),
                    "w": nir.Linear(weight=np.ones((2, 2))),
                    "d": nir.Delay(delay=np.full(2, 2.0)),
                },
                [("a", "w"), ("w", "d"), ("d", "b"), ("d", "c")],
                "w: its output reaches , which is not a population",
            ),
            (
                {"a": _if(2), "w": nir.Linear(weight=np.ones((2, 2))), "o": _output(2)},
                [("a", "w"), ("w", "o")],
                "w: its output reaches o, which is not a population",
            ),
            (
                {"a": _if(2), "b": _if(2), "w": nir.Linear(weight=np.eye(2))},
                [("a", "w"), ("w", "b")],
                "w: the Studio graph holds only all-to-all projections",
            ),
            (
                {"a": _if(2), "b": _if(2), "w": nir.Linear(weight=np.zeros((2, 2)))},
                [("a", "w"), ("w", "b")],
                "w: the Studio graph holds only all-to-all projections",
            ),
            (
                {"a": _if(2001)},
                [],
                "the imported network does not validate",
            ),
        ],
    )
    def test_what_the_studio_graph_cannot_hold_is_refused(self, nodes, edges, reason):
        with pytest.raises(NIRMappingRefused, match=reason):
            nir_file_to_graph(_foreign(nodes, edges))


class TestUnreadableInput:
    @pytest.mark.parametrize("payload", [None, "", 12, b"abc"])
    def test_an_import_needs_base64_text(self, payload):
        with pytest.raises(NIRMappingRefused, match="needs the file's bytes as base64 text"):
            nir_file_to_graph(payload)

    def test_text_that_is_not_base64_is_refused(self):
        with pytest.raises(NIRMappingRefused, match="not valid base64"):
            nir_file_to_graph("not*base64!")

    def test_bytes_that_are_not_nir_are_refused(self):
        with pytest.raises(NIRMappingRefused, match="not a readable NIR graph"):
            nir_file_to_graph(base64.b64encode(b"plain text, not HDF5").decode("ascii"))

    def test_an_hdf5_file_that_is_not_nir_is_refused(self, tmp_path):
        import h5py

        path = tmp_path / "other.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("values", data=np.arange(3))
        with pytest.raises(NIRMappingRefused, match="not a readable NIR graph"):
            nir_file_to_graph(base64.b64encode(path.read_bytes()).decode("ascii"))
