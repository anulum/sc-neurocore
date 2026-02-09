"""Accelerated SCPN components."""

from __future__ import annotations

import numpy as np

from sc_neurocore_engine.sc_neurocore_engine import KuramotoSolver as _RustKuramoto


class KuramotoSolver:
    """
    Drop-in replacement for the Kuramoto coupling loop in
    L4_CellularLayer and the UPDE solver.
    """

    def __init__(self, omega, coupling, phases, noise_amp=0.1):
        self._engine = _RustKuramoto(
            np.asarray(omega, dtype=np.float64).tolist(),
            np.asarray(coupling, dtype=np.float64).ravel().tolist(),
            np.asarray(phases, dtype=np.float64).tolist(),
            float(noise_amp),
        )

    def step(self, dt: float, seed: int = 0) -> float:
        return float(self._engine.step(float(dt), int(seed)))

    def run(self, n_steps: int, dt: float, seed: int = 0) -> np.ndarray:
        return np.array(self._engine.run(int(n_steps), float(dt), int(seed)), dtype=np.float64)

    def order_parameter(self) -> float:
        return float(self._engine.order_parameter())

    @property
    def phases(self) -> np.ndarray:
        return np.array(self._engine.get_phases(), dtype=np.float64)

    @phases.setter
    def phases(self, new_phases):
        self._engine.set_phases(np.asarray(new_phases, dtype=np.float64).tolist())
