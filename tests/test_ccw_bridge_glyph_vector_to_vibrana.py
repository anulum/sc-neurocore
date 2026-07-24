# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGlyphVectorToVibrana from former test_ccw_bridge.py

"""Focused suite: TestGlyphVectorToVibrana from former test_ccw_bridge.py."""

from __future__ import annotations

from tests.ccw_bridge_support import *  # noqa: F403


class TestGlyphVectorToVibrana:
    def _full(self, phi=0.0, fib=0.0, metatron=0.0, platonic=0.0, e8=0.0, health=0.0):
        return np.array([phi, fib, metatron, platonic, e8, health])

    def test_short_vector_is_padded(self):
        bridge = create_bridge()
        out = bridge.glyph_vector_to_vibrana(np.array([0.5, 0.5]))
        # Padding to 6 entries means the missing components read as zero.
        assert out["glyph_weights"]["symbolic_health"] == pytest.approx(0.0)
        assert bridge.vibrana_state.glyph_weights.shape == (6,)

    def test_mode_theurgic_on_high_metatron(self):
        bridge = create_bridge()
        out = bridge.glyph_vector_to_vibrana(self._full(metatron=0.8))
        assert out["mode"] == "theurgic"

    def test_mode_cosmic_on_phi_and_fibonacci(self):
        bridge = create_bridge()
        out = bridge.glyph_vector_to_vibrana(self._full(phi=0.9, fib=0.9, metatron=0.1))
        assert out["mode"] == "cosmic"

    def test_mode_healing_on_symbolic_health(self):
        bridge = create_bridge()
        out = bridge.glyph_vector_to_vibrana(self._full(health=0.7))
        assert out["mode"] == "healing"

    def test_mode_meditation_on_e8(self):
        bridge = create_bridge()
        out = bridge.glyph_vector_to_vibrana(self._full(e8=0.8))
        assert out["mode"] == "meditation"

    def test_mode_focus_is_the_fallback(self):
        bridge = create_bridge()
        out = bridge.glyph_vector_to_vibrana(self._full(platonic=0.2))
        assert out["mode"] == "focus"

    def test_output_structure_and_frequencies(self):
        bridge = create_bridge()
        out = bridge.glyph_vector_to_vibrana(self._full(metatron=0.8, health=0.4))
        # Geometry phase is wrapped into [0, 2π).
        assert 0.0 <= out["geometry_phase"] < 2 * np.pi
        assert out["color_intensity"] == pytest.approx(0.4)
        # rotation_speed = 0.5 + metatron * 2.0
        assert out["rotation_speed"] == pytest.approx(0.5 + 0.8 * 2.0)
        assert set(out["glyph_weights"]) == {
            "phi_alignment",
            "fibonacci_alignment",
            "metatron_flow",
            "platonic_coherence",
            "e8_alignment",
            "symbolic_health",
        }
        # Frequencies come from the selected mode's MODE_FREQUENCIES entry.
        assert out["frequencies"]["base"] == pytest.approx(7.83)
        assert out["frequencies"]["harmonic"] == pytest.approx(14.3)

    def test_geometry_phase_accumulates_across_calls(self):
        bridge = create_bridge()
        first = bridge.glyph_vector_to_vibrana(self._full(platonic=1.0))
        second = bridge.glyph_vector_to_vibrana(self._full(platonic=1.0))
        # Each call adds platonic_coherence * 0.1 to the running phase.
        assert second["geometry_phase"] == pytest.approx(first["geometry_phase"] + 0.1)
