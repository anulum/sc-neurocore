# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the CCW/VIBRANA bridge

"""Behavioural tests for the SC-NeuroCore ↔ CCW/VIBRANA bridge.

The bridge is a pure data transformation layer (stdlib + numpy, no live CCW
system): it maps SCPN layer metrics onto binaural-audio parameters, L7 glyph
vectors onto VIBRANA visualisation states, and packages both into metadata /
session configs. These tests exercise every mapping, mode-selection branch,
smoothing path, glyph-length guard, and the optional file-export sink.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sc_neurocore.interfaces.ccw_bridge import (
    CCWBridge,
    CCWMode,
    CCWParameters,
    VIBRANAState,
    create_bridge,
)


class TestDefaults:
    def test_ccw_parameters_defaults(self):
        p = CCWParameters()
        assert p.base_frequency == pytest.approx(7.83)  # Schumann
        assert p.carrier_frequency == pytest.approx(432.0)  # Verdi A4
        assert p.binaural_offset == pytest.approx(10.0)
        assert p.modulation_depth == pytest.approx(0.5)
        assert p.sample_rate == 44100

    def test_vibrana_state_defaults(self):
        s = VIBRANAState()
        assert s.mode is CCWMode.MEDITATION
        assert s.geometry_phase == 0.0
        # glyph_weights default_factory produces a zeroed 6-vector.
        assert s.glyph_weights.shape == (6,)
        assert np.all(s.glyph_weights == 0.0)

    def test_ccw_mode_string_values(self):
        assert CCWMode.THEURGIC.value == "theurgic"
        assert {m.value for m in CCWMode} == {
            "theurgic",
            "healing",
            "meditation",
            "cosmic",
            "focus",
            "creativity",
        }

    def test_init_uses_default_parameters_when_none(self):
        bridge = CCWBridge()
        assert isinstance(bridge.params, CCWParameters)
        assert bridge.params.sample_rate == 44100
        assert bridge.vibrana_state.mode is CCWMode.MEDITATION
        assert bridge.smoothing_window == 10

    def test_init_accepts_explicit_parameters(self):
        params = CCWParameters(carrier_frequency=528.0, sample_rate=22050)
        bridge = CCWBridge(params)
        assert bridge.params is params
        assert bridge.params.carrier_frequency == pytest.approx(528.0)


class TestBitstreamToFrequency:
    def test_all_ones_maps_to_max(self):
        bridge = create_bridge()
        assert bridge.bitstream_to_frequency(np.ones(8)) == pytest.approx(40.0)

    def test_all_zeros_maps_to_min(self):
        bridge = create_bridge()
        assert bridge.bitstream_to_frequency(np.zeros(8)) == pytest.approx(1.0)

    def test_half_density_maps_to_midpoint(self):
        bridge = create_bridge()
        bits = np.array([1, 0, 1, 0])
        assert bridge.bitstream_to_frequency(bits) == pytest.approx(20.5)

    def test_custom_range(self):
        bridge = create_bridge()
        # prob = 0.25 -> 100 + 0.25 * (200 - 100) = 125
        bits = np.array([1, 0, 0, 0])
        assert bridge.bitstream_to_frequency(bits, freq_min=100.0, freq_max=200.0) == pytest.approx(
            125.0
        )


class TestScpnMetricsToCcw:
    def test_no_metrics_returns_defaults(self):
        bridge = create_bridge()
        params = bridge.scpn_metrics_to_ccw({})
        # All eight keys present at their neutral defaults.
        assert params["amplitude"] == pytest.approx(0.5)
        assert params["carrier_blend"] == pytest.approx(0.5)
        assert params["schumann_blend"] == pytest.approx(0.5)
        assert params["sacred_geometry_intensity"] == pytest.approx(0.5)
        assert params["binaural_offset"] == pytest.approx(10.0)

    def test_mapped_metric_scaled_into_range(self):
        bridge = create_bridge()
        # l4_cellular_sync -> binaural_offset in [4, 40]; value 1.0 -> 40.
        params = bridge.scpn_metrics_to_ccw({"l4_cellular_sync": 1.0})
        assert params["binaural_offset"] == pytest.approx(40.0)

    def test_mapped_metric_lower_bound(self):
        bridge = create_bridge()
        params = bridge.scpn_metrics_to_ccw({"l4_cellular_sync": 0.0})
        assert params["binaural_offset"] == pytest.approx(4.0)

    def test_partial_metrics_leave_others_at_default(self):
        bridge = create_bridge()
        params = bridge.scpn_metrics_to_ccw({"l1_quantum_coherence": 1.0})
        # modulation_depth mapped into [0.3, 0.8] -> 0.8; carrier_blend untouched.
        assert params["modulation_depth"] == pytest.approx(0.8)
        assert params["carrier_blend"] == pytest.approx(0.5)

    def test_history_is_smoothed_and_window_bounded(self):
        bridge = create_bridge()
        # Feed 15 samples of a single metric; the history must cap at the window.
        for _ in range(15):
            bridge.scpn_metrics_to_ccw({"l1_quantum_coherence": 0.5})
        assert len(bridge.metric_history["l1_quantum_coherence"]) == bridge.smoothing_window

    def test_smoothing_averages_recent_values(self):
        bridge = create_bridge()
        bridge.scpn_metrics_to_ccw({"l6_planetary_coherence": 0.0})
        params = bridge.scpn_metrics_to_ccw({"l6_planetary_coherence": 1.0})
        # schumann_blend in [0, 1]; smoothed mean of {0.0, 1.0} = 0.5.
        assert params["schumann_blend"] == pytest.approx(0.5)


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


class TestGenerateBinauralSample:
    def test_shapes_match_requested_duration(self):
        bridge = create_bridge()
        left, right = bridge.generate_binaural_sample({"carrier_frequency": 432.0}, 256)
        assert left.shape == (256,)
        assert right.shape == (256,)

    def test_default_duration_is_1024(self):
        bridge = create_bridge()
        left, right = bridge.generate_binaural_sample({})
        assert left.shape == (1024,)
        assert right.shape == (1024,)

    def test_amplitude_bounds_the_signal(self):
        bridge = create_bridge()
        amplitude = 0.3
        left, right = bridge.generate_binaural_sample({"amplitude": amplitude}, 512)
        assert np.max(np.abs(left)) <= amplitude + 1e-9
        assert np.max(np.abs(right)) <= amplitude + 1e-9

    def test_phase_state_is_continuous(self):
        bridge = create_bridge()
        bridge.generate_binaural_sample({"carrier_frequency": 100.0}, 128)
        first_phase = bridge.phase_left
        bridge.generate_binaural_sample({"carrier_frequency": 100.0}, 128)
        # Phase advances and is wrapped into [0, 2π).
        assert 0.0 <= bridge.phase_left < 2 * np.pi
        assert bridge.phase_left != pytest.approx(first_phase)


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


class TestCreateSessionConfig:
    def test_default_meditation_session(self):
        bridge = create_bridge()
        cfg = bridge.create_session_config()
        assert cfg["session"]["mode"] == "meditation"
        assert cfg["session"]["duration_minutes"] == 20
        # MODE_FREQUENCIES[MEDITATION] = (4.0, 7.83)
        assert cfg["audio"]["base_frequency"] == pytest.approx(4.0)
        assert cfg["audio"]["harmonic_frequency"] == pytest.approx(7.83)
        assert cfg["scpn_integration"]["enabled"] is True

    def test_explicit_mode_and_duration(self):
        bridge = create_bridge()
        cfg = bridge.create_session_config(mode=CCWMode.COSMIC, duration_minutes=45)
        assert cfg["session"]["mode"] == "cosmic"
        assert cfg["session"]["duration_minutes"] == 45
        # MODE_FREQUENCIES[COSMIC] = (136.1, 272.2) (OM)
        assert cfg["audio"]["base_frequency"] == pytest.approx(136.1)
        assert cfg["audio"]["harmonic_frequency"] == pytest.approx(272.2)
        assert cfg["visual"]["color_scheme"] == "cosmic"


class TestFactory:
    def test_create_bridge_without_params(self):
        bridge = create_bridge()
        assert isinstance(bridge, CCWBridge)
        assert isinstance(bridge.params, CCWParameters)

    def test_create_bridge_with_params(self):
        params = CCWParameters(binaural_offset=6.0)
        bridge = create_bridge(params)
        assert bridge.params.binaural_offset == pytest.approx(6.0)
