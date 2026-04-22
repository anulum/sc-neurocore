# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Accelerated SCPN components

"""Accelerated SCPN components."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sc_neurocore_engine.sc_neurocore_engine import KuramotoSolver as _RustKuramoto


class KuramotoSolver:
    """
    Drop-in replacement for the Kuramoto coupling loop in
    L4_CellularLayer and the UPDE solver.
    """

    def __init__(
        self,
        omega: npt.ArrayLike,
        coupling: npt.ArrayLike,
        phases: npt.ArrayLike,
        noise_amp: float = 0.1,
    ) -> None:
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

    def set_field_pressure(self, f: float) -> None:
        """Set the external field pressure strength F."""
        self._engine.set_field_pressure(float(f))

    def step_ssgf(
        self,
        dt: float,
        seed: int = 0,
        W: np.ndarray | None = None,
        sigma_g: float = 0.0,
        h_munu: np.ndarray | None = None,
        pgbo_weight: float = 0.0,
    ) -> float:
        """SSGF-compatible step with geometry and PGBO coupling."""
        w_arg = np.asarray(W, dtype=np.float64) if W is not None else None
        h_arg = np.asarray(h_munu, dtype=np.float64) if h_munu is not None else None
        return float(
            self._engine.step_ssgf(
                float(dt),
                int(seed),
                w_arg,
                float(sigma_g),
                h_arg,
                float(pgbo_weight),
            )
        )

    def run_ssgf(
        self,
        n_steps: int,
        dt: float,
        seed: int = 0,
        W: np.ndarray | None = None,
        sigma_g: float = 0.0,
        h_munu: np.ndarray | None = None,
        pgbo_weight: float = 0.0,
    ) -> np.ndarray:
        """Run N SSGF-compatible steps."""
        w_arg = np.asarray(W, dtype=np.float64) if W is not None else None
        h_arg = np.asarray(h_munu, dtype=np.float64) if h_munu is not None else None
        return np.array(
            self._engine.run_ssgf(
                int(n_steps),
                float(dt),
                int(seed),
                w_arg,
                float(sigma_g),
                h_arg,
                float(pgbo_weight),
            ),
            dtype=np.float64,
        )

    def order_parameter(self) -> float:
        return float(self._engine.order_parameter())

    @property
    def phases(self) -> np.ndarray:
        getter = getattr(self._engine, "get_phases", None)
        if callable(getter):
            return np.array(getter(), dtype=np.float64)
        return np.array(self._engine.phases, dtype=np.float64)

    @phases.setter
    def phases(self, new_phases: npt.ArrayLike) -> None:
        self._engine.set_phases(np.asarray(new_phases, dtype=np.float64).tolist())
