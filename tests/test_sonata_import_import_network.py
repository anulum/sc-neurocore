# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImportNetwork from former test_sonata_import.py

"""Focused suite: TestImportNetwork from former test_sonata_import.py."""

from __future__ import annotations

from tests.sonata_import_support import *  # noqa: F403

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
