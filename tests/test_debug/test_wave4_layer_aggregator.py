# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayerAggregator from former test_wave4.py

"""Focused suite: TestLayerAggregator from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403

class TestLayerAggregator:
    def test_record_and_get(self):
        la = LayerAggregator()
        la.record(SpikeEvent(layer_id="L0", correlation=0.1, precision=0.95))
        la.record(SpikeEvent(layer_id="L0", correlation=0.3, precision=0.85))
        ls = la.get("L0")
        assert ls is not None
        assert ls["event_count"] == 2
        assert la.mean_correlation(ls) == pytest.approx(0.2)

    def test_missing_layer(self):
        la = LayerAggregator()
        assert la.get("missing") is None
