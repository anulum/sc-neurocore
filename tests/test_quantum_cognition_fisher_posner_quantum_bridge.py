# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFisherPosnerQuantumBridge from former test_quantum_cognition.py

"""Focused suite: TestFisherPosnerQuantumBridge from former test_quantum_cognition.py."""

from __future__ import annotations

from tests.quantum_cognition_support import *  # noqa: F403

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
