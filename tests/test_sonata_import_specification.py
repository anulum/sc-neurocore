# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SONATA population identity, property groups and refusals

"""SONATA files read through libsonata keep identity, groups and absences as stated."""

from __future__ import annotations

from collections.abc import Callable

from tests.sonata_import_support import *  # noqa: F403


def _two_population_nodes(path: Path) -> Path:
    """An ``exc`` and an ``inh`` population whose ids both start at 0."""
    with h5py.File(path, "w") as f:
        _stamp(f)
        for name, size, model in (("exc", 3, "point_neuron"), ("inh", 2, "virtual")):
            grp = f.create_group(f"nodes/{name}")
            grp.create_dataset("node_type_id", data=np.full(size, 1 if name == "exc" else 2))
            grp.create_dataset("node_group_id", data=np.zeros(size, dtype=np.uint32))
            grp.create_dataset("node_group_index", data=np.arange(size, dtype=np.uint64))
            grp.create_group("0").create_dataset(
                "model_type", data=np.array([model] * size, dtype=h5py.string_dtype())
            )
    return path


class TestPopulationIdentity:
    def test_nodes_are_identified_by_population_and_id(self, tmp_path: Path) -> None:
        nodes = _two_population_nodes(tmp_path / "nodes.h5")
        edges = _create_edges_h5(
            tmp_path / "edges.h5",
            src_ids=[2, 0],
            tgt_ids=[1, 0],
            weights=[0.5, -1.25],
            pop_name="exc_to_inh",
            source="exc",
            target="inh",
            delays=[1.5, 2.0],
        )

        net = import_sonata(nodes, edges)

        assert net.node_populations == {"exc": [0, 1, 2], "inh": [0, 1]}
        assert net.edge_populations == {"exc_to_inh": [0, 1]}
        assert [(node.population, node.node_id, node.model_type) for node in net.nodes] == [
            ("exc", 0, "point_neuron"),
            ("exc", 1, "point_neuron"),
            ("exc", 2, "point_neuron"),
            ("inh", 0, "virtual"),
            ("inh", 1, "virtual"),
        ]
        first = net.edges[0]
        assert (
            first.source_population,
            first.source_id,
            first.target_population,
            first.target_id,
        ) == (
            "exc",
            2,
            "inh",
            1,
        )
        assert (first.weight, first.delay) == (0.5, 1.5)
        matrix = net.connectivity_matrix()
        assert matrix[4, 2] == 0.5  # inh:1 <- exc:2
        assert matrix[3, 0] == -1.25  # inh:0 <- exc:0
        assert np.count_nonzero(matrix) == 2
        assert net.metadata == {
            "sonata_version": [0, 1],
            "edges_without_weight": 0,
            "edges_without_delay": 0,
        }

    def test_parallel_edges_add_their_weights(self, tmp_path: Path) -> None:
        nodes = _create_nodes_h5(tmp_path / "nodes.h5", n=2)
        edges = _create_edges_h5(
            tmp_path / "edges.h5", src_ids=[0, 0], tgt_ids=[1, 1], weights=[0.25, 0.5]
        )
        assert import_sonata(nodes, edges).connectivity_matrix()[1, 0] == 0.75


class TestProperties:
    def test_each_node_carries_its_group_properties_and_no_guessed_model(
        self, tmp_path: Path
    ) -> None:
        path = _create_nodes_h5(
            tmp_path / "nodes.h5", n=2, properties={"tau_m": np.array([10.0, 30.0])}
        )
        nodes = import_sonata_nodes(path)
        assert [node.properties for node in nodes] == [{"tau_m": 10.0}, {"tau_m": 30.0}]
        assert all(node.model_type is None and node.model_template is None for node in nodes)

    def test_edge_properties_other_than_weight_and_delay_are_kept(self, tmp_path: Path) -> None:
        path = _create_edges_h5(tmp_path / "edges.h5", src_ids=[0], tgt_ids=[1], weights=[2.0])
        with h5py.File(path, "a") as f:
            f["edges/exc_exc/0"].create_dataset(
                "syn_class", data=np.array(["ampa"], dtype=h5py.string_dtype())
            )
        (edge,) = import_sonata_edges(path)
        assert edge.properties == {"syn_class": "ampa"}


class TestRefusals:
    def test_a_population_spread_over_several_groups_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "nodes.h5"
        with h5py.File(path, "w") as f:
            _stamp(f)
            grp = f.create_group("nodes/mixed")
            grp.create_dataset("node_type_id", data=np.array([5, 6]))
            grp.create_dataset("node_group_id", data=np.array([0, 1], dtype=np.uint32))
            grp.create_dataset("node_group_index", data=np.array([0, 0], dtype=np.uint64))
            grp.create_group("0").create_dataset("tau_m", data=np.array([10.0]))
            grp.create_group("1").create_dataset("v_th", data=np.array([-50.0]))
        with pytest.raises(ValueError, match="spans several property groups"):
            import_sonata_nodes(path)

    def test_an_attribute_libsonata_cannot_read_is_an_error_not_a_gap(self, tmp_path: Path) -> None:
        path = _create_nodes_h5(
            tmp_path / "nodes.h5", n=1, properties={"model_type": np.array([b"point_neuron"])}
        )
        with pytest.raises(
            ValueError, match="attribute 'model_type' of population 'exc' cannot be read"
        ):
            import_sonata_nodes(path)

    def test_a_file_without_nodes_or_populations_is_refused(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare.h5"
        with h5py.File(bare, "w") as f:
            _stamp(f)
        with pytest.raises(ValueError, match="has no /nodes group"):
            import_sonata_nodes(bare)
        empty = tmp_path / "empty.h5"
        with h5py.File(empty, "w") as f:
            _stamp(f)
            f.create_group("nodes")
        with pytest.raises(ValueError, match="declares no node populations"):
            import_sonata_nodes(empty)

    def test_a_wrong_magic_is_refused(self, tmp_path: Path) -> None:
        path = _create_nodes_h5(tmp_path / "nodes.h5", n=1)
        with h5py.File(path, "a") as f:
            f.attrs["magic"] = np.uint32(7)
        with pytest.raises(ValueError, match="magic attribute is missing or wrong"):
            import_sonata(path)

    @pytest.mark.parametrize(
        ("group", "dataset"), [("nodes", "node_type_id"), ("edges", "edge_type_id")]
    )
    def test_missing_type_ids_are_refused(self, tmp_path: Path, group: str, dataset: str) -> None:
        reader: Callable[[Path], object]
        if group == "nodes":
            path = _create_nodes_h5(tmp_path / "f.h5", n=2)
            population, reader = "exc", import_sonata_nodes
        else:
            path = _create_edges_h5(tmp_path / "f.h5", src_ids=[0], tgt_ids=[1], weights=[1.0])
            population, reader = "exc_exc", import_sonata_edges
        with h5py.File(path, "a") as f:
            del f[f"{group}/{population}/{dataset}"]
        with pytest.raises(ValueError, match=f"has no {dataset} dataset"):
            reader(path)

    def test_type_ids_that_do_not_cover_the_population_are_refused(self, tmp_path: Path) -> None:
        path = _create_edges_h5(
            tmp_path / "edges.h5", src_ids=[0, 1, 0], tgt_ids=[1, 0, 1], weights=[1.0] * 3
        )
        with h5py.File(path, "a") as f:
            del f["edges/exc_exc/edge_type_id"]
            f["edges/exc_exc"].create_dataset("edge_type_id", data=np.array([1, 2]))
        with pytest.raises(ValueError, match="edge_type_id has 2 entries for 3 members"):
            import_sonata_edges(path)

    @pytest.mark.parametrize(
        ("source", "src_id", "message"),
        [("exc", 9, "names node 9 of population 'exc'"), ("other", 0, "population 'other'")],
    )
    def test_edges_naming_absent_nodes_are_refused(
        self, tmp_path: Path, source: str, src_id: int, message: str
    ) -> None:
        nodes = _create_nodes_h5(tmp_path / "nodes.h5", n=2)
        edges = _create_edges_h5(
            tmp_path / "edges.h5", src_ids=[src_id], tgt_ids=[1], weights=[1.0], source=source
        )
        with pytest.raises(ValueError, match=message):
            import_sonata(nodes, edges)

    def test_a_matrix_is_refused_while_any_weight_is_unstated(self, tmp_path: Path) -> None:
        nodes = _create_nodes_h5(tmp_path / "nodes.h5", n=2)
        edges = _create_edges_h5(tmp_path / "edges.h5", src_ids=[0], tgt_ids=[1])
        net = import_sonata(nodes, edges)
        assert net.metadata["edges_without_weight"] == 1
        with pytest.raises(ValueError, match="1 SONATA edge\\(s\\) state no syn_weight"):
            net.connectivity_matrix()

    def test_a_matrix_refuses_an_edge_to_a_node_outside_it(self, tmp_path: Path) -> None:
        nodes = _create_nodes_h5(tmp_path / "nodes.h5", n=2)
        edges = _create_edges_h5(tmp_path / "edges.h5", src_ids=[0], tgt_ids=[1], weights=[1.0])
        net = import_sonata(nodes, edges)
        net.nodes.pop()
        with pytest.raises(ValueError, match="names a node outside the network"):
            net.connectivity_matrix()
