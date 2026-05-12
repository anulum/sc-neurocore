# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L1 Quantum Biological Layer (Stochastic Implementation)

"""SCPN L1: Quantum Biological Layer (Stochastic Implementation)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Optional

import numpy as np

try:
    from sc_neurocore.quantum.hybrid import QuantumStochasticLayer
    from sc_neurocore.quantum.hardware_bridge import QuantumHardwareLayer

    _HAS_QUANTUM = True
except ImportError:
    _HAS_QUANTUM = False

logger = logging.getLogger(__name__)


@dataclass
class L1_StochasticParameters:
    """Parameters for the Stochastic L1 Layer."""

    n_qubits: int = 1000  # Number of simulated microtubules/dipoles
    bitstream_length: int = 1024

    # Biological Enhancements (from Legacy RAG)
    F_non_Markov: float = 1e4  # Protection factor
    temperature: float = 310.0  # human body temperature (K)

    # Quantum Dynamics
    coupling_strength: float = 0.1
    decoherence_rate: float = 0.05

    # Execution Backend
    # "simulated", "qiskit.aer_simulator", "pennylane.default.qubit"
    backend: str = "simulated"
    rng_seed: Optional[int] = None


class L1_QuantumLayer:
    """
    Stochastic implementation of the Quantum Cellular Field.
    """

    def __init__(self, params: Optional[L1_StochasticParameters] = None) -> None:
        self.params = params or L1_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)

        # The Core Engine
        if self.params.backend == "simulated":
            self.quantum_core: Any = QuantumStochasticLayer(
                n_qubits=self.params.n_qubits,
                length=self.params.bitstream_length,
                rng_seed=self.params.rng_seed,
            )
        else:
            self.quantum_core = QuantumHardwareLayer(
                n_qubits=self.params.n_qubits,
                length=self.params.bitstream_length,
                backend_type=self.params.backend,
            )

        # State: Coherence represented as probabilities (0.0 to 1.0)
        # In SC, this will be encoded into bitstreams.
        # Initialize with max coherence (pure state)
        self.coherence_probs = np.ones(self.params.n_qubits) * 0.95

        # History for non-Markovian effects
        self.history: list[float] = []

    def step(
        self, dt: float, external_field: Optional[np.ndarray[Any, Any]] = None
    ) -> np.ndarray[Any, Any]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            external_field: Optional coupling input from other layers (normalized 0-1).

        Returns:
            output_bitstreams: The stochastic state of the field.
        """
        self._validate_step_inputs(dt, external_field, self.params.n_qubits)
        # 1. Apply Decoherence (Classical Decay)
        # Adjusted by Non-Markovian factor
        effective_decay = self.params.decoherence_rate * dt / np.log10(self.params.F_non_Markov)
        self.coherence_probs *= 1.0 - effective_decay

        # 2. Apply External Coupling (e.g. from L2 Neurochemical)
        if external_field is not None:
            field = np.asarray(external_field, dtype=np.float64).reshape(self.params.n_qubits)
            # Mix the field: coherence is modulated by external input
            self.coherence_probs = (
                1 - self.params.coupling_strength
            ) * self.coherence_probs + self.params.coupling_strength * field

        # 3. Quantum Rotation via Stochastic Core
        # The core takes the probabilities, rotates them (simulating evolution),
        # and returns collapsed bitstreams.
        # We assume the 'probability' maps to the quantum phase/amplitude.

        # Generate input bitstreams from current probabilities
        rands = self._rng.random((self.params.n_qubits, self.params.bitstream_length))
        input_bits = (rands < self.coherence_probs[:, None]).astype(np.uint8)

        # Pass through Quantum Hybrid Layer
        output_bits = self.quantum_core.forward(input_bits)

        # 4. Update State from Measurement (Collapse/Update)
        # The output bits represent the measured state.
        # We update our internal probabilities based on the measurement (Bayesian update or similar)
        # For this simulation, we'll take the mean as the new base, but add some "Quantum Zeno" recovery
        measured_probs = np.mean(output_bits, axis=1)

        # "Zeno" effect: frequent measurement can freeze evolution or reset it.
        # Here we just blend it back.
        self.coherence_probs = 0.9 * self.coherence_probs + 0.1 * measured_probs

        res: np.ndarray[Any, Any] = output_bits
        return res

    def get_global_metric(self) -> float:
        """Return the global coherence metric (Phi-like)."""
        return float(np.mean(self.coherence_probs))

    @staticmethod
    def _validate_params(params: L1_StochasticParameters) -> None:
        if (
            not isinstance(params.n_qubits, int)
            or isinstance(params.n_qubits, bool)
            or params.n_qubits <= 0
        ):
            raise ValueError("n_qubits must be a positive integer")
        if (
            not isinstance(params.bitstream_length, int)
            or isinstance(params.bitstream_length, bool)
            or params.bitstream_length <= 0
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if not math.isfinite(float(params.F_non_Markov)) or params.F_non_Markov <= 1.0:
            raise ValueError("F_non_Markov must be finite and greater than 1")
        if not math.isfinite(float(params.temperature)) or params.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if (
            not math.isfinite(float(params.coupling_strength))
            or params.coupling_strength < 0.0
            or params.coupling_strength > 1.0
        ):
            raise ValueError("coupling_strength must be finite and within [0, 1]")
        if not math.isfinite(float(params.decoherence_rate)) or params.decoherence_rate < 0.0:
            raise ValueError("decoherence_rate must be finite and non-negative")
        if not isinstance(params.backend, str) or not params.backend:
            raise ValueError("backend must be a non-empty string")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @staticmethod
    def _validate_step_inputs(
        dt: float, external_field: Optional[np.ndarray[Any, Any]], n_qubits: int
    ) -> None:
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if external_field is not None:
            field = np.asarray(external_field, dtype=np.float64)
            if field.size != n_qubits or not np.all(np.isfinite(field)):
                raise ValueError("external_field must contain one finite value per qubit")
            if np.any(field < 0.0) or np.any(field > 1.0):
                raise ValueError("external_field must be within [0, 1]")
