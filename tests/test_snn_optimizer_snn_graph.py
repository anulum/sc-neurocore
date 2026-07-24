# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSNNGraph from former test_snn_optimizer.py

"""Focused suite: TestSNNGraph from former test_snn_optimizer.py."""

from __future__ import annotations

from tests.snn_optimizer_support import *  # noqa: F403


class TestSNNGraph:
    def test_total_params(self):
        g = _make_graph()
        assert g.total_params == 80 + 32 + 8

    def test_total_neurons(self):
        g = _make_graph()
        assert g.total_neurons == 14

    def test_copy(self):
        g = _make_graph()
        c = g.copy()
        c.layers[0].weights[0, 0] = 999
        assert g.layers[0].weights[0, 0] != 999
