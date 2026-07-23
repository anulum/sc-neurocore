# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCToPhotonic from former test_bridges_photonic_noc.py

"""Focused suite: TestSCToPhotonic from former test_bridges_photonic_noc.py."""

from __future__ import annotations

from tests.bridges_photonic_noc_support import *  # noqa: F403

class TestSCToPhotonic:
    def test_compile_simple_network(self):
        compiler = SCToPhotonic()
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        design = compiler.compile(adj)
        assert isinstance(design, PhotonicCircuitDesign)
        assert design.n_nodes >= 3
        assert len(design.waveguides) >= 1
        assert len(design.mzi_gates) >= 1

    def test_compile_larger_network(self):
        compiler = SCToPhotonic()
        rng = np.random.default_rng(42)
        adj = (rng.random((10, 10)) > 0.7).astype(float)
        np.fill_diagonal(adj, 0)
        design = compiler.compile(adj)
        assert design.n_nodes == 10

    def test_design_has_wdm_channels(self):
        compiler = SCToPhotonic()
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        design = compiler.compile(adj)
        assert len(design.wdm_channels) >= 1
