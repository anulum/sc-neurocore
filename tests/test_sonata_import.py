# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SONATA network format importer

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from sc_neurocore.adapters.sonata import (
    import_sonata,
    import_sonata_edges,
    import_sonata_nodes,
)


def _create_nodes_h5(path, n=10, pop_name="exc"):
    """Create a minimal SONATA nodes HDF5 file."""
    with h5py.File(path, "w") as f:
        grp = f.create_group(f"nodes/{pop_name}")
        grp.create_dataset("node_id", data=np.arange(n))
        grp.create_dataset("node_type_id", data=np.zeros(n, dtype=int))
    return path


def _create_edges_h5(path, src_ids, tgt_ids, weights=None, pop_name="exc_exc"):
    """Create a minimal SONATA edges HDF5 file."""
    with h5py.File(path, "w") as f:
        grp = f.create_group(f"edges/{pop_name}")
        grp.create_dataset("source_node_id", data=np.array(src_ids))
        grp.create_dataset("target_node_id", data=np.array(tgt_ids))
        grp.create_dataset("edge_type_id", data=np.zeros(len(src_ids), dtype=int))
        if weights is not None:
            g0 = grp.create_group("0")
            g0.create_dataset("syn_weight", data=np.array(weights))
    return path


class TestImportNodes:
    def test_basic(self, tmp_path):
        f = _create_nodes_h5(tmp_path / "nodes.h5", n=5)
        nodes = import_sonata_nodes(f)
        assert len(nodes) == 5
        assert nodes[0].node_id == 0
        assert nodes[4].node_id == 4

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.h5"
        with h5py.File(p, "w") as f:
            pass
        nodes = import_sonata_nodes(p)
        assert len(nodes) == 0


class TestImportEdges:
    def test_basic(self, tmp_path):
        f = _create_edges_h5(
            tmp_path / "edges.h5",
            src_ids=[0, 1, 2],
            tgt_ids=[3, 4, 5],
            weights=[0.5, 0.3, 0.8],
        )
        edges = import_sonata_edges(f)
        assert len(edges) == 3
        assert edges[0].source_id == 0
        assert edges[0].target_id == 3
        assert edges[0].weight == pytest.approx(0.5)

    def test_no_weights(self, tmp_path):
        f = _create_edges_h5(
            tmp_path / "edges.h5",
            src_ids=[0, 1],
            tgt_ids=[2, 3],
        )
        edges = import_sonata_edges(f)
        assert edges[0].weight == 1.0  # default


class TestImportNetwork:
    def test_full_network(self, tmp_path):
        nf = _create_nodes_h5(tmp_path / "nodes.h5", n=5)
        ef = _create_edges_h5(
            tmp_path / "edges.h5",
            src_ids=[0, 1, 2],
            tgt_ids=[3, 4, 1],
            weights=[0.5, 0.3, 0.8],
        )
        net = import_sonata(nf, ef)
        assert net.n_nodes == 5
        assert net.n_edges == 3

    def test_connectivity_matrix(self, tmp_path):
        nf = _create_nodes_h5(tmp_path / "nodes.h5", n=3)
        ef = _create_edges_h5(
            tmp_path / "edges.h5",
            src_ids=[0, 1],
            tgt_ids=[1, 2],
            weights=[0.5, 0.8],
        )
        net = import_sonata(nf, ef)
        W = net.connectivity_matrix()
        assert W.shape == (3, 3)
        assert W[1, 0] == pytest.approx(0.5)
        assert W[2, 1] == pytest.approx(0.8)
        assert W[0, 0] == 0.0

    def test_nodes_only(self, tmp_path):
        nf = _create_nodes_h5(tmp_path / "nodes.h5", n=10)
        net = import_sonata(nf)
        assert net.n_nodes == 10
        assert net.n_edges == 0
