# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L11 Morphic Resonance / Noospheric Layer

"""
SCPN L11: Morphic Resonance / Noospheric Layer (Stochastic Implementation)

Ising-style spin-glass with memetic SIR dynamics for cultural/informational
field evolution.

H = -sum(J_ij σ_i σ_j) - sum(h_i σ_i)  (NTHS Hamiltonian)
dS/dt = -β S I / N  (memetic SIR)

Ref: Paper 11 — Noosphere-Technosphere Hybrid System.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L11_StochasticParameters:
    n_nodes: int = 100
    bitstream_length: int = 1024
    j_coupling: float = 0.5
    h_bias: float = 0.1
    beta_infection: float = 0.2
    gamma_recovery: float = 0.05
    boundary_coupling: float = 0.1  # from L10
    boundary_shielding_coupling: float = 0.35
    boundary_pressure_coupling: float = 0.25
    rng_seed: Optional[int] = None


class L11_MorphicLayer:
    """Noospheric spin-glass with memetic spreading dynamics."""

    def __init__(self, params: Optional[L11_StochasticParameters] = None):
        self.params = params or L11_StochasticParameters()
        self._validate_params(self.params)
        n = self.params.n_nodes
        self.spins = np.full(n, 0.5)
        self.info_density = np.zeros(n)
        self.time = 0.0
        self._rng = np.random.default_rng(self.params.rng_seed)

    def step(
        self,
        dt: float,
        l10_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time += dt
        n = self.params.n_nodes

        boundary_shielding = 0.0
        boundary_fragmentation_pressure = 0.0
        l10_rejection_fraction = 0.0
        field_input = np.zeros(n)
        if l10_input is not None:
            l10_effect = self._l10_boundary_effect(l10_input, n)
            boundary_shielding = l10_effect["shielding"]
            boundary_fragmentation_pressure = l10_effect["fragmentation_pressure"]
            l10_rejection_fraction = l10_effect["rejection_fraction"]
            protected_drive = l10_effect["integrity"] * self.params.boundary_coupling
            fragmented_drive = (
                boundary_fragmentation_pressure * self.params.boundary_pressure_coupling
            )
            field_input = np.full(n, protected_drive - fragmented_drive)

        mean_field = np.mean(self.spins)
        d_spin = (
            self.params.j_coupling * mean_field
            + self.params.h_bias
            + field_input
            - 0.1 * self.spins
        )
        self.spins = np.clip(self.spins + d_spin * dt, 0, 1)
        transmission = max(
            0.0, 1.0 - self.params.boundary_shielding_coupling * boundary_shielding
        )
        self.info_density = self._update_info_density(dt, transmission)

        rands = self._rng.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < self.spins[:, None]).astype(np.uint8)

        return {
            "spins": self.spins.copy(),
            "polarization": float(np.std(self.spins)),
            "info_saturation": float(np.mean(self.info_density)),
            "boundary_shielding": boundary_shielding,
            "boundary_fragmentation_pressure": boundary_fragmentation_pressure,
            "l10_rejection_fraction": l10_rejection_fraction,
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        return float(np.mean(self.spins))

    @staticmethod
    def _validate_params(params: L11_StochasticParameters) -> None:
        if not isinstance(params.n_nodes, int) or isinstance(params.n_nodes, bool):
            raise ValueError("n_nodes must be a positive integer")
        if params.n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if not math.isfinite(float(params.j_coupling)):
            raise ValueError("j_coupling must be finite")
        if not math.isfinite(float(params.h_bias)):
            raise ValueError("h_bias must be finite")
        if not math.isfinite(float(params.beta_infection)) or params.beta_infection < 0.0:
            raise ValueError("beta_infection must be finite and non-negative")
        if not math.isfinite(float(params.gamma_recovery)) or params.gamma_recovery < 0.0:
            raise ValueError("gamma_recovery must be finite and non-negative")
        if not math.isfinite(float(params.boundary_coupling)) or params.boundary_coupling < 0.0:
            raise ValueError("boundary_coupling must be finite and non-negative")
        if (
            not math.isfinite(float(params.boundary_shielding_coupling))
            or params.boundary_shielding_coupling < 0.0
        ):
            raise ValueError("boundary_shielding_coupling must be finite and non-negative")
        if (
            not math.isfinite(float(params.boundary_pressure_coupling))
            or params.boundary_pressure_coupling < 0.0
        ):
            raise ValueError("boundary_pressure_coupling must be finite and non-negative")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @staticmethod
    def _integrity_signal(value: Any) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError("integrity must be a finite scalar")
        integrity = float(values)
        if not math.isfinite(integrity):
            raise ValueError("integrity must be a finite scalar")
        return integrity

    @classmethod
    def _l10_boundary_effect(cls, l10_input: Dict[str, Any], n_nodes: int) -> Dict[str, float]:
        integrity = cls._integrity_signal(l10_input.get("integrity", 0.0))
        if integrity < 0.0 or integrity > 1.0:
            raise ValueError("integrity must be within [0, 1]")

        firewall_strength = cls._project_nonnegative_vector(
            l10_input.get("firewall_strength", np.zeros(n_nodes)),
            n_nodes,
            "firewall_strength",
        )
        topological_rejection = cls._project_rejection_mask(
            l10_input.get("topological_rejection_mask", np.zeros(n_nodes, dtype=bool)),
            n_nodes,
        )
        boundary_complexity = cls._nonnegative_scalar(
            l10_input.get("boundary_complexity", 0.0), "boundary_complexity"
        )
        qec_residual_load = cls._unit_scalar(
            l10_input.get("qec_residual_load", 0.0), "qec_residual_load"
        )
        memory_complexity_flux = cls._nonnegative_scalar(
            l10_input.get("memory_complexity_flux", 0.0), "memory_complexity_flux"
        )

        firewall_pressure = float(np.mean(firewall_strength / (1.0 + firewall_strength)))
        rejection_fraction = float(np.mean(topological_rejection))
        shielding = float(
            np.clip(
                firewall_pressure
                + rejection_fraction
                + boundary_complexity
                + qec_residual_load
                + memory_complexity_flux,
                0.0,
                1.0,
            )
        )
        fragmentation_pressure = float(
            boundary_complexity + qec_residual_load + memory_complexity_flux + rejection_fraction
        )
        return {
            "integrity": integrity,
            "shielding": shielding,
            "fragmentation_pressure": fragmentation_pressure,
            "rejection_fraction": rejection_fraction,
        }

    @staticmethod
    def _project_nonnegative_vector(value: Any, n_nodes: int, name: str) -> np.ndarray:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError(f"{name} must contain at least one value")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(values < 0.0):
            raise ValueError(f"{name} must be non-negative")
        if values.size >= n_nodes:
            return values[:n_nodes].copy()
        return np.pad(values, (0, n_nodes - values.size), constant_values=0.0)

    @staticmethod
    def _project_rejection_mask(value: Any, n_nodes: int) -> np.ndarray:
        values = np.asarray(value)
        if values.size == 0:
            raise ValueError("topological_rejection_mask must contain at least one value")
        if values.dtype == np.bool_:
            mask = values.reshape(-1)
        else:
            numeric = np.asarray(value, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(numeric)):
                raise ValueError("topological_rejection_mask must contain only finite values")
            if np.any((numeric != 0.0) & (numeric != 1.0)):
                raise ValueError("topological_rejection_mask must be boolean or 0/1")
            mask = numeric.astype(bool)
        if mask.size >= n_nodes:
            return mask[:n_nodes].copy()
        return np.pad(mask, (0, n_nodes - mask.size), constant_values=False)

    @staticmethod
    def _nonnegative_scalar(value: Any, name: str) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError(f"{name} must be a finite scalar")
        scalar = float(values)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar

    @classmethod
    def _unit_scalar(cls, value: Any, name: str) -> float:
        scalar = cls._nonnegative_scalar(value, name)
        if scalar > 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
        return scalar

    def _update_info_density(self, dt: float, transmission: float) -> np.ndarray:
        memetic_activation = np.clip(2.0 * np.abs(self.spins - 0.5), 0.0, 1.0)
        susceptible = 1.0 - self.info_density
        infection = self.params.beta_infection * transmission * susceptible * memetic_activation
        recovery = self.params.gamma_recovery * self.info_density
        return np.clip(self.info_density + (infection - recovery) * dt, 0.0, 1.0)
