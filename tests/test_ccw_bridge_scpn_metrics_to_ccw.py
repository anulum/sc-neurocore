# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScpnMetricsToCcw from former test_ccw_bridge.py

"""Focused suite: TestScpnMetricsToCcw from former test_ccw_bridge.py."""

from __future__ import annotations

import pytest
from sc_neurocore.interfaces.ccw_bridge import (
    create_bridge,
)


class TestScpnMetricsToCcw:
    """Mapping SCPN metrics to CCW parameters."""

    def test_no_metrics_returns_defaults(self) -> None:
        """No metrics give the default parameters."""
        bridge = create_bridge()
        params = bridge.scpn_metrics_to_ccw({})
        # All eight keys present at their neutral defaults.
        assert params["amplitude"] == pytest.approx(0.5)
        assert params["carrier_blend"] == pytest.approx(0.5)
        assert params["schumann_blend"] == pytest.approx(0.5)
        assert params["sacred_geometry_intensity"] == pytest.approx(0.5)
        assert params["binaural_offset"] == pytest.approx(10.0)

    def test_mapped_metric_scaled_into_range(self) -> None:
        """A top metric maps to the top of its range."""
        bridge = create_bridge()
        # l4_cellular_sync -> binaural_offset in [4, 40]; value 1.0 -> 40.
        params = bridge.scpn_metrics_to_ccw({"l4_cellular_sync": 1.0})
        assert params["binaural_offset"] == pytest.approx(40.0)

    def test_mapped_metric_lower_bound(self) -> None:
        """A bottom metric maps to the bottom of its range."""
        bridge = create_bridge()
        params = bridge.scpn_metrics_to_ccw({"l4_cellular_sync": 0.0})
        assert params["binaural_offset"] == pytest.approx(4.0)

    def test_partial_metrics_leave_others_at_default(self) -> None:
        """Unmapped parameters keep their defaults."""
        bridge = create_bridge()
        params = bridge.scpn_metrics_to_ccw({"l1_quantum_coherence": 1.0})
        # modulation_depth mapped into [0.3, 0.8] -> 0.8; carrier_blend untouched.
        assert params["modulation_depth"] == pytest.approx(0.8)
        assert params["carrier_blend"] == pytest.approx(0.5)

    def test_history_is_smoothed_and_window_bounded(self) -> None:
        """The metric history is bounded by its window."""
        bridge = create_bridge()
        # Feed 15 samples of a single metric; the history must cap at the window.
        for _ in range(15):
            bridge.scpn_metrics_to_ccw({"l1_quantum_coherence": 0.5})
        assert len(bridge.metric_history["l1_quantum_coherence"]) == bridge.smoothing_window

    def test_smoothing_averages_recent_values(self) -> None:
        """Smoothing averages the recent values."""
        bridge = create_bridge()
        bridge.scpn_metrics_to_ccw({"l6_planetary_coherence": 0.0})
        params = bridge.scpn_metrics_to_ccw({"l6_planetary_coherence": 1.0})
        # schumann_blend in [0, 1]; smoothed mean of {0.0, 1.0} = 0.5.
        assert params["schumann_blend"] == pytest.approx(0.5)
