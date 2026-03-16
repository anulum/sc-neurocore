# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HeronR2NoiseParams:
    """IBM Heron r2 calibration parameters (2024)."""

    cx_error: float = 0.005
    single_qubit_error: float = 0.0003
    t1_us: float = 300.0
    t2_us: float = 200.0
    readout_0to1: float = 0.01
    readout_1to0: float = 0.02
    gate_time_1q_ns: float = 25.0
    gate_time_2q_ns: float = 100.0


class HeronR2NoiseModel:
    def __init__(self, params=None):
        self.params = params or HeronR2NoiseParams()

    def depolarizing_channel(self, p):
        """Kraus operators for single-qubit depolarizing channel."""
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return [
            np.sqrt(1 - p) * I,
            np.sqrt(p / 3) * X,
            np.sqrt(p / 3) * Y,
            np.sqrt(p / 3) * Z,
        ]

    def amplitude_damping(self, gamma):
        """Kraus operators for amplitude damping (T1 decay)."""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        return [K0, K1]

    def phase_damping(self, gamma):
        """Kraus operators for phase damping (T2 decay)."""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
        return [K0, K1]

    def apply_single_qubit_noise(self, rho):
        """Apply single-qubit noise channel to density matrix."""
        kraus = self.depolarizing_channel(self.params.single_qubit_error)
        return sum(K @ rho @ K.conj().T for K in kraus)

    def apply_readout_noise(self, measurement):
        """Apply asymmetric readout error."""
        p = self.params
        if measurement == 0:
            return 1 if np.random.random() < p.readout_0to1 else 0
        return 0 if np.random.random() < p.readout_1to0 else 1

    def gate_fidelity_1q(self):
        return 1.0 - self.params.single_qubit_error

    def gate_fidelity_2q(self):
        return 1.0 - self.params.cx_error
