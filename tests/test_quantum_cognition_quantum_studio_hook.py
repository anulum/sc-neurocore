# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantumStudioHook from former test_quantum_cognition.py

"""Focused suite: TestQuantumStudioHook from former test_quantum_cognition.py."""

from __future__ import annotations

from tests.quantum_cognition_support import *  # noqa: F403


class TestQuantumStudioHook:
    """Tests for the Studio visualisation hook."""

    @pytest.fixture
    def hook(self) -> QuantumStudioHook:
        pool = SpinPoolMPS(n_sites=4)
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        return QuantumStudioHook(pool, bridge)

    def test_layer_metadata(self, hook: QuantumStudioHook) -> None:
        meta = hook.get_layer_metadata()
        assert isinstance(meta, QuantumCognitionLayerMetadata)
        assert meta.layer_name == "Quantum Cognition (Fisher-Posner)"
        assert meta.n_sites == 4

    def test_layer_metadata_dict(self, hook: QuantumStudioHook) -> None:
        d = hook.get_layer_metadata_dict()
        assert d["layer_name"] == "Quantum Cognition (Fisher-Posner)"
        assert "metrics" in d
        assert "visual_config" in d
        assert d["visual_config"]["color"] == "#00f2ff"

    def test_realtime_data(self, hook: QuantumStudioHook) -> None:
        data = hook.get_realtime_data()
        assert len(data["entanglement_map"]) == 4
        assert len(data["atp_efficiencies"]) == 4
        assert data["bridge_backend"] == "emulated"

    def test_entanglement_snapshot_payload(self, hook: QuantumStudioHook) -> None:
        snapshot = hook.get_entanglement_snapshot()
        assert snapshot["n_sites"] == 4
        assert len(snapshot["entanglement_map"]) == 4
        assert len(snapshot["atp_efficiencies"]) == 4
        assert snapshot["measurement_count"] == 0
        assert snapshot["coherence_status"] == "stable"
        assert snapshot["bridge_backend"] == "emulated"
        assert isinstance(snapshot["timestamp"], float)

    def test_json_event_is_compact_ndjson_payload(self, hook: QuantumStudioHook) -> None:
        payload = hook.to_json_event("quantum_snapshot")
        assert "\n" not in payload

        event = json.loads(payload)
        assert event["event"] == "quantum_snapshot"
        assert event["timestamp"] == event["data"]["timestamp"]
        assert event["data"]["n_sites"] == 4
        assert event["data"]["bridge_backend"] == "emulated"

    def test_repr(self, hook: QuantumStudioHook) -> None:
        rendered = repr(hook)
        assert "QuantumStudioHook" in rendered
        assert "SpinPoolMPS" in rendered
        assert "FisherPosnerQuantumBridge" in rendered
