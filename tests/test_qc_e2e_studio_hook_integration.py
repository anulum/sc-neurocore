# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStudioHookIntegration from former test_qc_e2e.py

"""Focused suite: TestStudioHookIntegration from former test_qc_e2e.py."""

from __future__ import annotations

from tests.qc_e2e_support import *  # noqa: F403


class TestStudioHookIntegration:
    """Verify telemetry hook produces valid structured data."""

    def test_snapshot_structure(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        bridge = FisherPosnerQuantumBridge(n_qubits=4, backend="emulated")
        hook = QuantumStudioHook(pool, bridge)

        pool.apply_measurement(3, 1.0)
        snap = hook.get_entanglement_snapshot()
        assert "timestamp" in snap
        assert snap["n_sites"] == 8
        assert len(snap["entanglement_map"]) == 8
        assert len(snap["atp_efficiencies"]) == 8

    def test_json_event_valid(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        bridge = FisherPosnerQuantumBridge(n_qubits=4, backend="emulated")
        hook = QuantumStudioHook(pool, bridge)

        event_str = hook.to_json_event("test_event")
        event = json.loads(event_str)
        assert event["event"] == "test_event"
        assert "data" in event

    def test_dashboard_no_crash(self) -> None:
        """Dashboard should render without crashing."""
        brain = GOTMBrain(n_neurons=8, seed=42)
        chunk = ContentChunk("t", "t.md", 0, "content", "markdown", 1.0)
        brain.learn_step(chunk, np.ones(8) * 0.5)

        dashboard = TerminalDashboard(clear_screen=False)
        # Should not raise
        dashboard.draw(brain)
