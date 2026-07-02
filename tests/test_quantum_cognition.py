# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for quantum_cognition subpackage

"""Comprehensive tests for the quantum cognition (Fisher-Posner) layer.

Covers: SpinPoolMPS, HybridFisherPosnerLIF, FisherPosnerQuantumBridge,
QuantumStudioHook, and cross-module non-locality verification.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from sc_neurocore.quantum_cognition.spin_pool import SpinCouplingTensor, SpinPoolMPS
from sc_neurocore.quantum_cognition.fisher_posner import HybridFisherPosnerLIF
from sc_neurocore.quantum_cognition.bridge_adapter import (
    FisherPosnerQuantumBridge,
    HAS_PENNYLANE,
)
from sc_neurocore.quantum_cognition.studio_hook import (
    QuantumStudioHook,
    QuantumCognitionLayerMetadata,
)


# ───────── SpinPoolMPS ─────────


class TestSpinPoolMPS:
    """Tests for the Matrix Product States spin pool emulator."""

    def test_init_defaults(self) -> None:
        pool = SpinPoolMPS()
        assert pool.n_sites == 8
        assert pool.bond_dim == 16
        assert pool.entanglement_map.shape == (8,)
        assert np.isclose(np.sum(pool.entanglement_map), 1.0)

    def test_init_custom(self) -> None:
        pool = SpinPoolMPS(n_sites=4, bond_dim=8, correlation_length=3.0, update_rate=0.2)
        assert pool.n_sites == 4
        assert pool.bond_dim == 8
        assert pool.correlation_length == 3.0
        assert pool.update_rate == 0.2

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="n_sites"):
            SpinPoolMPS(n_sites=0)
        with pytest.raises(ValueError, match="bond_dim"):
            SpinPoolMPS(bond_dim=0)
        with pytest.raises(ValueError, match="correlation_length"):
            SpinPoolMPS(correlation_length=-1.0)
        with pytest.raises(ValueError, match="update_rate"):
            SpinPoolMPS(update_rate=0.0)
        with pytest.raises(ValueError, match="update_rate"):
            SpinPoolMPS(update_rate=1.5)

    def test_apply_measurement_updates_map(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        initial_map = pool.entanglement_map.copy()
        pool.apply_measurement(3, 1.0)
        # After measurement at site 3, entanglement should concentrate near site 3
        assert not np.allclose(pool.entanglement_map, initial_map)
        assert np.isclose(np.sum(pool.entanglement_map), 1.0)
        assert pool._measurement_count == 1

    def test_measurement_site_bounds(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        with pytest.raises(IndexError, match="site_idx"):
            pool.apply_measurement(-1)
        with pytest.raises(IndexError, match="site_idx"):
            pool.apply_measurement(4)
        with pytest.raises(ValueError, match="intensity"):
            pool.apply_measurement(0, intensity=-0.5)

    def test_atp_efficiency_range(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        for i in range(8):
            eff = pool.get_local_atp_efficiency(i)
            assert 0.0 <= eff <= 1.0, f"Efficiency {eff} out of range at site {i}"

    def test_atp_efficiency_site_bounds(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        with pytest.raises(IndexError, match="site_idx"):
            pool.get_local_atp_efficiency(4)

    def test_repeated_measurements_concentrate_entanglement(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        for _ in range(50):
            pool.apply_measurement(0, 1.0)
        # Entanglement should be highest at site 0 after repeated spikes there
        assert pool.entanglement_map[0] > pool.entanglement_map[7]

    def test_get_status(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        pool.apply_measurement(1)
        status = pool.get_status()
        assert status["n_sites"] == 4
        assert status["measurement_count"] == 1
        assert "avg_entanglement" in status
        assert status["coherence_status"] == "stable"

    def test_get_set_state_roundtrip(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        pool.apply_measurement(2, 0.5)
        state = pool.get_state()
        pool2 = SpinPoolMPS(n_sites=4)
        pool2.set_state(state)
        np.testing.assert_array_almost_equal(pool.entanglement_map, pool2.entanglement_map)

    def test_reset(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        pool.apply_measurement(0)
        pool.apply_measurement(1)
        pool.reset()
        assert pool._measurement_count == 0
        assert np.allclose(pool.entanglement_map, 0.25)

    def test_to_scpn_payload(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        payload = pool.to_scpn_payload()
        assert "quantum_cognition_spin_pool" in payload
        inner = payload["quantum_cognition_spin_pool"]
        assert inner["n_sites"] == 4
        assert len(inner["entanglement_map"]) == 4
        assert len(inner["atp_efficiencies"]) == 4

    def test_repr(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        r = repr(pool)
        assert "SpinPoolMPS" in r
        assert "n_sites=4" in r


# ───────── HybridFisherPosnerLIF ─────────


class TestHybridFisherPosnerLIF:
    """Tests for the quantum-metabolic LIF neuron."""

    @pytest.fixture
    def pool(self) -> SpinPoolMPS:
        return SpinPoolMPS(n_sites=8)

    def test_init(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(0, pool)
        assert n.Vm == -70.0
        assert n.atp_level == 1.0
        assert n.id == 0

    def test_init_validation(self, pool: SpinPoolMPS) -> None:
        with pytest.raises(ValueError, match="neuron_id"):
            HybridFisherPosnerLIF(-1, pool)
        with pytest.raises(ValueError, match="exceeds"):
            HybridFisherPosnerLIF(99, pool)
        with pytest.raises(TypeError, match="SpinPoolMPS"):
            HybridFisherPosnerLIF(0, "not_a_pool")  # type: ignore[arg-type]

    def test_subthreshold_no_spike(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(0, pool)
        vm, spiked = n.step(0.0)
        assert not spiked
        assert vm < n.v_threshold

    def test_suprathreshold_spike(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(0, pool)
        # Large current should cause spike
        for _ in range(100):
            vm, spiked = n.step(50.0)
            if spiked:
                break
        assert n._total_spikes > 0

    def test_metabolic_failure(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(0, pool, atp_consumption=0.5)
        n.atp_level = 0.01  # Nearly depleted
        n.Vm = n.v_threshold + 5.0  # Above threshold
        vm, spiked = n.step(0.0)
        # Should fail to spike due to insufficient ATP
        assert not spiked
        assert n._metabolic_failures > 0

    def test_atp_regeneration(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(0, pool)
        n.atp_level = 0.5  # Partially depleted
        initial_atp = n.atp_level
        n.step(0.0)  # Subthreshold step should regenerate some ATP
        assert n.atp_level >= initial_atp

    def test_spike_feedback_to_spin_pool(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(0, pool)
        initial_count = pool._measurement_count
        # Drive neuron to spike
        for _ in range(200):
            n.step(50.0)
        # Spikes should have triggered measurements
        if n._total_spikes > 0:
            assert pool._measurement_count > initial_count

    def test_get_state(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(3, pool)
        n.step(10.0)
        state = n.get_state()
        assert state["neuron_id"] == 3
        assert "Vm" in state
        assert "atp_level" in state
        assert state["total_steps"] == 1

    def test_reset_state(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(0, pool)
        for _ in range(50):
            n.step(50.0)
        n.reset_state()
        assert n.Vm == n.v_rest
        assert n.atp_level == 1.0
        assert n._total_spikes == 0

    def test_repr(self, pool: SpinPoolMPS) -> None:
        n = HybridFisherPosnerLIF(2, pool)
        r = repr(n)
        assert "HybridFisherPosnerLIF" in r
        assert "id=2" in r


# ───────── Non-locality verification ─────────


class TestNonLocality:
    """Verify that quantum coupling produces non-local effects."""

    def test_explicit_hamiltonian_changes_distal_observable(self) -> None:
        """Non-local effects require explicit physical coupling tensors."""
        pool = SpinPoolMPS(n_sites=8, bond_dim=8)
        eff7_before = pool.get_local_atp_efficiency(7)
        tensor = np.zeros((3, 3), dtype=np.float64)
        tensor[0, 0] = 1.0
        pool.evolve_exact([SpinCouplingTensor(0, 7, tensor)], time_us=0.25)
        eff7_after = pool.get_local_atp_efficiency(7)
        assert eff7_before != eff7_after

    def test_proximal_stronger_than_distal(self) -> None:
        """Nearby neurons should be more affected than distant ones."""
        pool = SpinPoolMPS(n_sites=8)
        n0 = HybridFisherPosnerLIF(0, pool)

        # Record initial efficiencies
        for _ in range(100):
            n0.step(50.0)

        if n0._total_spikes > 0:
            eff_near = pool.entanglement_map[1]
            eff_far = pool.entanglement_map[7]
            assert eff_near > eff_far, (
                f"Proximity violation: near={eff_near:.4f}, far={eff_far:.4f}"
            )


# ───────── FisherPosnerQuantumBridge ─────────


class TestFisherPosnerQuantumBridge:
    """Tests for the quantum bridge adapter."""

    def test_init_emulated(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        assert bridge.n_qubits == 4
        assert bridge.backend == "emulated"
        assert bridge.dev is None

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="n_qubits"):
            FisherPosnerQuantumBridge(0)

    def test_sync_emulated(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        result = bridge.execute_non_local_sync([(0, 1), (2, 3)])
        assert result.shape == (4,)
        # Entangled qubits should have <Z>=0
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 0.0

    def test_sync_no_pairs(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        result = bridge.execute_non_local_sync([])
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, np.ones(4))

    def test_optimize_phases_emulated_returns_none(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        result = bridge.optimize_phases(0.5)
        assert result is None

    def test_orchestrator_bias_validation(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        with pytest.raises(ValueError, match="global_phases"):
            bridge.apply_orchestrator_bias(np.zeros(3), 0.5)

    def test_qpu_artifact_metadata(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        meta = bridge.to_qpu_artifact_metadata()
        assert meta["bridge_type"] == "FisherPosnerQuantumBridge"
        assert meta["n_qubits"] == 4
        assert meta["tier"] == "experimental"

    def test_repr(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="emulated")
        r = repr(bridge)
        assert "FisherPosnerQuantumBridge" in r

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_sync_pennylane(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="pennylane")
        result = bridge.execute_non_local_sync([(0, 1)])
        assert result.shape == (4,)

    @pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
    def test_optimize_phases_pennylane(self) -> None:
        bridge = FisherPosnerQuantumBridge(4, backend="pennylane")
        result = bridge.optimize_phases(0.5, n_steps=3)
        assert result is not None
        assert result.shape == (4,)


# ───────── QuantumStudioHook ─────────


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


# ───────── Package import ─────────


class TestPackageImport:
    """Verify the package-level imports work correctly."""

    def test_import_all(self) -> None:
        from sc_neurocore.quantum_cognition import (
            SpinPoolMPS,
            HybridFisherPosnerLIF,
            FisherPosnerQuantumBridge,
            QuantumStudioHook,
        )

        assert SpinPoolMPS is not None
        assert HybridFisherPosnerLIF is not None
        assert FisherPosnerQuantumBridge is not None
        assert QuantumStudioHook is not None

    def test_tier_label(self) -> None:
        from sc_neurocore import quantum_cognition

        assert quantum_cognition.__tier__ == "experimental"
