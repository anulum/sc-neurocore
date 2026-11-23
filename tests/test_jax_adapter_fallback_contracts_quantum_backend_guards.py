# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantumBackendGuards from former test_jax_adapter_fallback_contracts.py

"""Focused suite: TestQuantumBackendGuards from former test_jax_adapter_fallback_contracts.py."""

from __future__ import annotations

import pytest
from unittest.mock import (
    patch,
)


class TestQuantumBackendGuards:
    """Quantum backends without their libraries."""

    def test_aer_without_qiskit(self) -> None:
        """The Aer backend without Qiskit is rejected."""
        with patch("sc_neurocore.quantum.hardware_bridge.HAS_QISKIT", False):
            from sc_neurocore.quantum.hardware_bridge import QuantumHardwareLayer

            with pytest.raises(RuntimeError, match="Qiskit"):
                QuantumHardwareLayer(n_qubits=2, backend_type="aer_simulator")

    def test_pennylane_without_pennylane(self) -> None:
        """The PennyLane backend without PennyLane is rejected."""
        with patch("sc_neurocore.quantum.hardware_bridge.HAS_PENNYLANE", False):
            from sc_neurocore.quantum.hardware_bridge import QuantumHardwareLayer

            with pytest.raises(RuntimeError, match="PennyLane"):
                QuantumHardwareLayer(n_qubits=2, backend_type="pennylane.default.qubit")
