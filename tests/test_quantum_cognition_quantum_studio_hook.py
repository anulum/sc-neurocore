# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantumStudioHook from former test_quantum_cognition.py

"""Focused suite: TestQuantumStudioHook from former test_quantum_cognition.py."""

from __future__ import annotations

import json
import pytest
from sc_neurocore.quantum_cognition.bridge_adapter import (
    FisherPosnerQuantumBridge,
)
from sc_neurocore.quantum_cognition.spin_pool import (
    SpinPoolMPS,
)
from sc_neurocore.quantum_cognition.studio_hook import (
    QuantumCognitionLayerMetadata,
    QuantumStudioHook,
)


class TestQuantumStudioHook:
    """Tests for the Studio visualisation hook."""

    @pytest.fixture
    def hook(self) -> QuantumStudioHook:
        """Provide a studio hook over a four-site pool and an emulated bridge."""
        pool = SpinPoolMPS(n_sites=4)
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        return QuantumStudioHook(pool, bridge)

    def test_layer_metadata(self, hook: QuantumStudioHook) -> None:
        """The hook describes its Fisher-Posner layer."""
        meta = hook.get_layer_metadata()
        assert isinstance(meta, QuantumCognitionLayerMetadata)
        assert meta.layer_name == "Quantum Cognition (Fisher-Posner)"
        assert meta.n_sites == 4

    def test_layer_metadata_dict(self, hook: QuantumStudioHook) -> None:
        """The layer metadata converts to a dictionary with metrics."""
        d = hook.get_layer_metadata_dict()
        assert d["layer_name"] == "Quantum Cognition (Fisher-Posner)"
        assert "metrics" in d
        assert "visual_config" in d
        assert d["visual_config"]["color"] == "#00f2ff"

    def test_realtime_data(self, hook: QuantumStudioHook) -> None:
        """Real-time data has one value per site."""
        data = hook.get_realtime_data()
        assert len(data["entanglement_map"]) == 4
        assert len(data["atp_efficiencies"]) == 4
        assert data["bridge_backend"] == "emulated"

    def test_entanglement_snapshot_payload(self, hook: QuantumStudioHook) -> None:
        """The entanglement snapshot has one value per site."""
        snapshot = hook.get_entanglement_snapshot()
        assert snapshot["n_sites"] == 4
        assert len(snapshot["entanglement_map"]) == 4
        assert len(snapshot["atp_efficiencies"]) == 4
        assert snapshot["measurement_count"] == 0
        assert snapshot["coherence_status"] == "stable"
        assert snapshot["bridge_backend"] == "emulated"
        assert isinstance(snapshot["timestamp"], float)

    def test_json_event_is_compact_ndjson_payload(self, hook: QuantumStudioHook) -> None:
        """The JSON event is one compact NDJSON line."""
        payload = hook.to_json_event("quantum_snapshot")
        assert "\n" not in payload

        event = json.loads(payload)
        assert event["event"] == "quantum_snapshot"
        assert event["timestamp"] == event["data"]["timestamp"]
        assert event["data"]["n_sites"] == 4
        assert event["data"]["bridge_backend"] == "emulated"

    def test_repr(self, hook: QuantumStudioHook) -> None:
        """The representation names the hook and its parts."""
        rendered = repr(hook)
        assert "QuantumStudioHook" in rendered
        assert "SpinPoolMPS" in rendered
        assert "FisherPosnerQuantumBridge" in rendered
