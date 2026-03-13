# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Hardware Bridge for Quantum-Classical Hybrid execution.

This module provides the interface to offload the simulated quantum
stochastic logic to actual quantum hardware via Qiskit, or to high-fidelity
tensor-network simulators via PennyLane.

Usage::

    from sc_neurocore.quantum.hardware_bridge import QuantumHardwareLayer
    layer = QuantumHardwareLayer(n_qubits=4, backend_type="aer_simulator")
    out_bits = layer.forward(input_bitstreams)
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    import qiskit  # noqa: F401
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


@dataclass
class QuantumHardwareLayer:
    """
    Executes a Quantum-Classical Hybrid Layer on Qiskit/PennyLane.
    Maps bitstream probability -> Qubit Rotation -> True Measurement.
    """

    n_qubits: int
    length: int = 1024
    backend_type: str = "aer_simulator"  # "aer_simulator", "pennylane.default.qubit", etc.
    _qiskit_simulator: Any = None
    _pennylane_dev: Any = None

    def __post_init__(self) -> None:
        if self.backend_type.startswith("aer") and not HAS_QISKIT:
            from sc_neurocore.exceptions import SCDependencyError

            raise SCDependencyError("Qiskit is required for aer_simulator backend.")
        if self.backend_type.startswith("pennylane") and not HAS_PENNYLANE:
            from sc_neurocore.exceptions import SCDependencyError

            raise SCDependencyError("PennyLane is required for pennylane backend.")

        if self.backend_type == "aer_simulator":
            self._qiskit_simulator = AerSimulator()
        elif self.backend_type.startswith("pennylane"):
            self._pennylane_dev = qml.device(
                "default.qubit", wires=self.n_qubits, shots=self.length
            )

    def forward(self, input_bitstreams: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        input_bitstreams: (n_qubits, length)
        Returns:
        output_bitstreams: (n_qubits, length)
        """
        p_in = np.mean(input_bitstreams, axis=1)
        theta = p_in * np.pi

        if self.backend_type == "aer_simulator":
            return self._run_qiskit(theta)
        elif self.backend_type.startswith("pennylane"):
            return self._run_pennylane(theta)
        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")

    def _run_qiskit(self, theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Runs the circuit on Qiskit AerSimulator for `length` shots."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)

        # Apply Ry rotations based on theta
        for i in range(self.n_qubits):
            qc.ry(theta[i], i)

        qc.measure(range(self.n_qubits), range(self.n_qubits))

        # Run circuit for self.length shots
        compiled_circuit = transpile(qc, self._qiskit_simulator)
        job = self._qiskit_simulator.run(compiled_circuit, shots=self.length)
        result = job.result()
        counts = result.get_counts(compiled_circuit)

        # Reconstruct bitstreams from shot counts
        out_bits = np.zeros((self.n_qubits, self.length), dtype=np.uint8)
        current_idx = 0
        for bitstring, count in counts.items():
            # bitstring is like '0101' where index 0 is the last character in string
            for i in range(count):
                if current_idx < self.length:
                    for qubit_idx in range(self.n_qubits):
                        # Qiskit orders bitstrings right-to-left
                        bit_val = int(bitstring[self.n_qubits - 1 - qubit_idx])
                        # Invert because measurement logic expects |0> as 1
                        out_bits[qubit_idx, current_idx] = 1 - bit_val
                    current_idx += 1

        return out_bits

    def _run_pennylane(self, theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Runs the circuit on PennyLane for `length` shots."""

        @qml.qnode(self._pennylane_dev)  # type: ignore[untyped-decorator]
        def circuit(angles: np.ndarray[Any, Any]) -> Any:
            for i in range(self.n_qubits):
                qml.RY(angles[i], wires=i)
            return qml.sample()

        # Returns shape: (shots, n_qubits)
        samples = circuit(theta)
        # Transpose to (n_qubits, shots) and invert so |0> -> 1
        res: np.ndarray[Any, Any] = (1 - samples).T.astype(np.uint8)
        return res
