# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest
import numpy as np

from sc_neurocore.quantum.hardware_bridge import (
    QuantumHardwareLayer,
    HAS_QISKIT,
    HAS_PENNYLANE,
)


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit is not installed")
def test_qiskit_backend():
    layer = QuantumHardwareLayer(n_qubits=2, length=100, backend_type="aer_simulator")

    # Input probability 0.0 -> theta 0 -> cos(0) = 1 -> |0> -> bit 1
    # Input probability 1.0 -> theta pi -> cos(pi/2) = 0 -> |1> -> bit 0
    input_bits = np.zeros((2, 100), dtype=np.uint8)
    input_bits[0, :] = 0  # prob 0.0
    input_bits[1, :] = 1  # prob 1.0

    out_bits = layer.forward(input_bits)

    assert out_bits.shape == (2, 100)
    # Qubit 0 should mostly be 1 (since it was measured as |0>)
    assert np.mean(out_bits[0, :]) > 0.9
    # Qubit 1 should mostly be 0 (since it was measured as |1>)
    assert np.mean(out_bits[1, :]) < 0.1


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane is not installed")
def test_pennylane_backend():
    layer = QuantumHardwareLayer(n_qubits=2, length=100, backend_type="pennylane.default.qubit")

    input_bits = np.zeros((2, 100), dtype=np.uint8)
    input_bits[0, :] = 0
    input_bits[1, :] = 1

    out_bits = layer.forward(input_bits)

    assert out_bits.shape == (2, 100)
    assert np.mean(out_bits[0, :]) > 0.9
    assert np.mean(out_bits[1, :]) < 0.1
