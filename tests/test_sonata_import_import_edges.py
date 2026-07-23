# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImportEdges from former test_sonata_import.py

"""Focused suite: TestImportEdges from former test_sonata_import.py."""

from __future__ import annotations

from tests.sonata_import_support import *  # noqa: F403

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
