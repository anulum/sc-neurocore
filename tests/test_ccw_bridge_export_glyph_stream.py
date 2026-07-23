# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExportGlyphStream from former test_ccw_bridge.py

"""Focused suite: TestExportGlyphStream from former test_ccw_bridge.py."""

from __future__ import annotations

from tests.ccw_bridge_support import *  # noqa: F403

class TestExportGlyphStream:
    def test_full_vector_serialises_all_components(self):
        bridge = create_bridge()
        glyph = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        payload = json.loads(bridge.export_glyph_stream(glyph))
        gv = payload["glyph_vector"]
        assert gv["phi_alignment"] == pytest.approx(0.1)
        assert gv["symbolic_health"] == pytest.approx(0.6)
        assert payload["routing"]["target"] == "vibrana_hardware"
        assert payload["cosmic_vector"] == {}

    def test_short_vector_pads_missing_components_with_zero(self):
        bridge = create_bridge()
        payload = json.loads(bridge.export_glyph_stream(np.array([0.7, 0.8])))
        gv = payload["glyph_vector"]
        assert gv["phi_alignment"] == pytest.approx(0.7)
        assert gv["fibonacci_alignment"] == pytest.approx(0.8)
        assert gv["metatron_flow"] == pytest.approx(0.0)
        assert gv["symbolic_health"] == pytest.approx(0.0)

    def test_empty_vector_yields_all_zero_components(self):
        bridge = create_bridge()
        payload = json.loads(bridge.export_glyph_stream(np.array([])))
        assert all(v == pytest.approx(0.0) for v in payload["glyph_vector"].values())

    def test_cosmic_vector_is_included_when_provided(self):
        bridge = create_bridge()
        payload = json.loads(
            bridge.export_glyph_stream(np.zeros(6), cosmic_vector={"l8_phase": 0.25})
        )
        assert payload["cosmic_vector"] == {"l8_phase": 0.25}

    def test_writes_file_when_path_given(self, tmp_path):
        bridge = create_bridge()
        target = tmp_path / "glyph_stream.json"
        returned = bridge.export_glyph_stream(np.arange(6, dtype=float), filepath=str(target))
        assert target.exists()
        on_disk = json.loads(target.read_text())
        assert on_disk == json.loads(returned)
        assert on_disk["glyph_vector"]["phi_alignment"] == pytest.approx(0.0)

    def test_no_file_written_when_path_omitted(self, tmp_path):
        bridge = create_bridge()
        bridge.export_glyph_stream(np.zeros(6))
        assert list(tmp_path.iterdir()) == []
