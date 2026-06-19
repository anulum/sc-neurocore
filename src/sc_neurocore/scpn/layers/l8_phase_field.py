# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L8 Cosmic Phase-Locking Layer (Stochastic

"""SCPN L8: cosmic phase-locking layer (stochastic implementation).

Pulsar Timing Array synchronization: local oscillators phase-lock to
cosmic reference frequencies via Kuramoto coupling.

dTheta_n/dt = Omega_n + K_cosmic * sum(sin(Theta_m - Theta_n))

Ref: Paper 8 — Cosmic Phase-Locking and PTA synchronisation.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L8_StochasticParameters:
    """Stochastic configuration parameters for the SCPN cosmic phase-locking layer."""

    n_pulsars: int = 12
    bitstream_length: int = 1024
    k_cosmic: float = 0.05
    symbolic_coupling: float = 0.1  # from L7
    director_coupling: float = 0.15  # to L16
    pulsar_omegas: Optional[np.ndarray[Any, Any]] = None
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and finalise the layer parameters after construction."""
        if self.pulsar_omegas is None:
            base_omegas = np.array([1.6, 2.3, 0.8, 4.1, 1.1, 0.5, 3.2, 2.7, 1.9, 0.4, 5.5, 0.2])
            self.pulsar_omegas = np.resize(base_omegas, self.n_pulsars)


class L8_PhaseFieldLayer:
    """Stochastic cosmic phase-locking via Kuramoto-coupled PTA oscillators."""

    def __init__(self, params: Optional[L8_StochasticParameters] = None):
        self.params = params or L8_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)
        self.phases = self._rng.uniform(0, 2 * np.pi, self.params.n_pulsars)
        self.time = 0.0

    def step(
        self,
        dt: float,
        l7_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Advance the cosmic phase-locking layer one timestep and return its output state."""
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time += dt
        n = self.params.n_pulsars
        omegas = self.params.pulsar_omegas

        # Kuramoto coupling: phase differences
        phase_diff = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        coupling = self.params.k_cosmic * np.sum(np.sin(phase_diff), axis=1) / n

        d_phase = omegas + coupling
        if l7_input is not None:
            drive = self._l7_phase_drive(l7_input)
            d_phase += self.params.symbolic_coupling * drive * np.sin(-self.phases)

        self.phases = (self.phases + d_phase * dt) % (2 * np.pi)

        activation = (1.0 + np.cos(self.phases)) / 2.0
        rands = self._rng.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)
        memory_imprint_drive = self._memory_imprint_drive()
        cosmic_alignment = memory_imprint_drive["reference_amplitude"]

        return {
            "phases": self.phases.copy(),
            "cosmic_alignment": cosmic_alignment,
            "memory_imprint_drive": memory_imprint_drive,
            "director_drive": self.params.director_coupling * cosmic_alignment,
            "output_bitstreams": output_bitstreams,
        }

    def _order_parameter(self) -> float:
        return float(np.abs(np.mean(np.exp(1j * self.phases))))

    def _memory_imprint_drive(self) -> Dict[str, float]:
        reference = np.mean(np.exp(1j * self.phases))
        amplitude = float(np.abs(reference))
        phase = 0.0 if amplitude <= 1e-12 else float(np.angle(reference))
        return {
            "reference_amplitude": amplitude,
            "reference_phase": phase,
            "reference_real": float(amplitude * math.cos(phase)),
        }

    def get_global_metric(self) -> float:
        """Return the scalar global metric summarising this layer's state."""
        return self._order_parameter()

    @staticmethod
    def _validate_params(params: L8_StochasticParameters) -> None:
        if not isinstance(params.n_pulsars, int) or isinstance(params.n_pulsars, bool):
            raise ValueError("n_pulsars must be a positive integer")
        if params.n_pulsars <= 0:
            raise ValueError("n_pulsars must be positive")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if not math.isfinite(float(params.k_cosmic)) or params.k_cosmic < 0.0:
            raise ValueError("k_cosmic must be finite and non-negative")
        if not math.isfinite(float(params.symbolic_coupling)) or params.symbolic_coupling < 0.0:
            raise ValueError("symbolic_coupling must be finite and non-negative")
        if not math.isfinite(float(params.director_coupling)) or params.director_coupling < 0.0:
            raise ValueError("director_coupling must be finite and non-negative")
        if params.pulsar_omegas is None:
            raise ValueError("pulsar_omegas must be initialised")
        omegas = np.asarray(params.pulsar_omegas, dtype=np.float64).reshape(-1)
        if omegas.size != params.n_pulsars:
            raise ValueError("pulsar_omegas length must match n_pulsars")
        if not np.all(np.isfinite(omegas)):
            raise ValueError("pulsar_omegas must contain only finite values")
        params.pulsar_omegas = omegas
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @staticmethod
    def _glyph_drive(glyph_vector: Any) -> float:
        values = np.asarray(glyph_vector, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("glyph_vector must contain at least one value")
        if not np.all(np.isfinite(values)):
            raise ValueError("glyph_vector must contain only finite values")
        return float(np.mean(values))

    @classmethod
    def _l7_phase_drive(cls, l7_input: Dict[str, Any]) -> float:
        if "cosmic_phase_drive" in l7_input:
            return cls._nonnegative_scalar(l7_input["cosmic_phase_drive"], "cosmic_phase_drive")
        if "glyph_vector" in l7_input:
            return cls._glyph_drive(l7_input["glyph_vector"])
        return 0.0

    @staticmethod
    def _nonnegative_scalar(value: Any, name: str) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError(f"{name} must be a finite scalar")
        scalar = float(values)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar
