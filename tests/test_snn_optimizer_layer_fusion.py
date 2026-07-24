# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayerFusion from former test_snn_optimizer.py

"""Focused suite: TestLayerFusion from former test_snn_optimizer.py."""

from __future__ import annotations

from tests.snn_optimizer_support import *  # noqa: F403


class TestLayerFusion:
    def test_fuses_silent_layer(self):
        g = SNNGraph(
            layers=[
                LayerNode(
                    "h1", 4, 3, np.random.randn(3, 4), firing_rates=np.full(3, 0.001)
                ),  # silent → fusible
                LayerNode("h2", 3, 2, np.random.randn(2, 3), firing_rates=np.full(2, 0.1)),
            ]
        )
        result = layer_fusion(g)
        assert result.layers_fused == 1
        assert len(g.layers) == 1
        assert g.layers[0].weights.shape == (2, 4)

    def test_no_fusion_active_layers(self):
        g = _make_graph()
        result = layer_fusion(g)
        assert result.layers_fused == 0
