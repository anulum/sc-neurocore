# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGenerateCcwMetadata from former test_ccw_bridge.py

"""Focused suite: TestGenerateCcwMetadata from former test_ccw_bridge.py."""

from __future__ import annotations

from tests.ccw_bridge_support import *  # noqa: F403

class TestGenerateCcwMetadata:
    def test_extracts_numeric_coherence_metrics(self):
        bridge = create_bridge()
        outputs = {"l1": {"coherence": 0.5, "count": 3}}
        meta = bridge.generate_ccw_metadata(outputs)
        assert meta["scpn_metrics"]["l1_coherence"] == pytest.approx(0.5)
        assert meta["scpn_metrics"]["l1_count"] == pytest.approx(3.0)
        assert meta["bridge_version"] == "1.0.0"
        assert isinstance(meta["timestamp"], float)

    def test_non_dict_output_is_ignored(self):
        bridge = create_bridge()
        meta = bridge.generate_ccw_metadata({"l1": [1, 2, 3]})
        assert meta["scpn_metrics"] == {}

    def test_dict_without_coherence_key_is_ignored(self):
        bridge = create_bridge()
        meta = bridge.generate_ccw_metadata({"l1": {"activity": 0.9}})
        assert meta["scpn_metrics"] == {}

    def test_non_numeric_values_are_skipped(self):
        bridge = create_bridge()
        outputs = {"l1": {"coherence": 0.5, "label": "text"}}
        meta = bridge.generate_ccw_metadata(outputs)
        assert meta["scpn_metrics"] == {"l1_coherence": 0.5}

    def test_explicit_glyph_vector_populates_vibrana(self):
        bridge = create_bridge()
        glyph = np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.4])
        meta = bridge.generate_ccw_metadata({}, glyph_vector=glyph)
        assert meta["vibrana_visual"]["mode"] == "theurgic"
        assert meta["mode"] == "theurgic"

    def test_glyph_vector_recovered_from_l7_output(self):
        bridge = create_bridge()
        glyph = np.array([0.9, 0.9, 0.1, 0.0, 0.0, 0.0])
        meta = bridge.generate_ccw_metadata({"l7": {"glyph_vector": glyph}})
        assert meta["vibrana_visual"]["mode"] == "cosmic"

    def test_l7_without_glyph_vector_leaves_vibrana_empty(self):
        bridge = create_bridge()
        meta = bridge.generate_ccw_metadata({"l7": {"something_else": 1.0}})
        assert meta["vibrana_visual"] == {}

    def test_no_glyph_and_no_l7_leaves_vibrana_empty(self):
        bridge = create_bridge()
        meta = bridge.generate_ccw_metadata({"l1": {"coherence": 0.2}})
        assert meta["vibrana_visual"] == {}
