# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L12 Quantum Information / Gaian Layer (Stochastic

"""
SCPN L12: Quantum Information / Gaian Layer (Stochastic Implementation)

Environment-Assisted Quantum Transport (ENAQT) model for ecological
coherence across a network of sites.

S = -Tr(ρ ln ρ)  (von Neumann entropy)
Transport efficiency: population transfer under dephasing noise.

Ref: Paper 12 — Gaian ENAQT and ecological quantum coherence.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L12_StochasticParameters:
    n_sites: int = 100
    bitstream_length: int = 1024
    transport_rate: float = 0.3
    dephasing_gamma: float = 0.05
    morphic_coupling: float = 0.1  # from L11
    noospheric_entropy_coupling: float = 0.15
    rng_seed: Optional[int] = None


class L12_QuantumInfoLayer:
    """ENAQT-inspired ecological coherence transport."""

    def __init__(self, params: Optional[L12_StochasticParameters] = None):
        self.params = params or L12_StochasticParameters()
        self._validate_params(self.params)
        n = self.params.n_sites
        self._rng = np.random.default_rng(self.params.rng_seed)
        self.coherence = self._rng.uniform(0.3, 0.7, n)
        self.time = 0.0

    def step(
        self,
        dt: float,
        l11_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time += dt
        n = self.params.n_sites

        noospheric_entropy_load = 0.0
        gaian_stabilization_drive = 0.0
        effective_dephasing_gamma = self.params.dephasing_gamma
        boundary_context_id: Optional[str] = None
        boundary_terminals: tuple[str, ...] = ()
        gaian_terminal_set: tuple[str, ...] = ()
        gaian_terminal_bandwidth = 1.0
        if l11_input is not None:
            l11_effect = self._l11_noospheric_effect(l11_input)
            noospheric_entropy_load = l11_effect["entropy_load"]
            gaian_stabilization_drive = l11_effect["stabilization_drive"]
            boundary_context_id = l11_effect["boundary_context_id"]
            boundary_terminals = l11_effect["boundary_terminals"]
            gaian_terminal_set = l11_effect["gaian_terminal_set"]
            gaian_terminal_bandwidth = l11_effect["gaian_terminal_bandwidth"]
            effective_dephasing_gamma = self.params.dephasing_gamma * (
                1.0 + self.params.noospheric_entropy_coupling * noospheric_entropy_load
            )

        # Nearest-neighbour transport (ring topology)
        transport = np.roll(self.coherence, 1) - 2 * self.coherence + np.roll(self.coherence, -1)
        dephasing = -effective_dephasing_gamma * self.coherence
        self.coherence += (self.params.transport_rate * transport + dephasing) * dt

        if l11_input is not None:
            self.coherence += gaian_stabilization_drive * dt

        self.coherence = np.clip(self.coherence, 0, 1)

        entropy = self._von_neumann_entropy()

        rands = self._rng.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < self.coherence[:, None]).astype(np.uint8)

        return {
            "coherence": self.coherence.copy(),
            "entropy": entropy,
            "transport_efficiency": float(np.mean(self.coherence)),
            "noospheric_entropy_load": noospheric_entropy_load,
            "gaian_stabilization_drive": gaian_stabilization_drive,
            "effective_dephasing_gamma": effective_dephasing_gamma,
            "boundary_context_id": boundary_context_id,
            "boundary_terminals": boundary_terminals,
            "gaian_terminal_set": gaian_terminal_set,
            "gaian_terminal_bandwidth": gaian_terminal_bandwidth,
            "output_bitstreams": output_bitstreams,
        }

    def _von_neumann_entropy(self) -> float:
        p = self.coherence / (np.sum(self.coherence) + 1e-10)
        return float(-np.sum(p * np.log(p + 1e-10)))

    def get_global_metric(self) -> float:
        return float(np.mean(self.coherence))

    @staticmethod
    def _validate_params(params: L12_StochasticParameters) -> None:
        if not isinstance(params.n_sites, int) or isinstance(params.n_sites, bool):
            raise ValueError("n_sites must be a positive integer")
        if params.n_sites <= 0:
            raise ValueError("n_sites must be positive")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if not math.isfinite(float(params.transport_rate)) or params.transport_rate < 0.0:
            raise ValueError("transport_rate must be finite and non-negative")
        if not math.isfinite(float(params.dephasing_gamma)) or params.dephasing_gamma < 0.0:
            raise ValueError("dephasing_gamma must be finite and non-negative")
        if not math.isfinite(float(params.morphic_coupling)) or params.morphic_coupling < 0.0:
            raise ValueError("morphic_coupling must be finite and non-negative")
        if (
            not math.isfinite(float(params.noospheric_entropy_coupling))
            or params.noospheric_entropy_coupling < 0.0
        ):
            raise ValueError("noospheric_entropy_coupling must be finite and non-negative")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @staticmethod
    def _info_saturation(value: Any) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError("info_saturation must be a finite scalar")
        info_saturation = float(values)
        if not math.isfinite(info_saturation) or info_saturation < 0.0 or info_saturation > 1.0:
            raise ValueError("info_saturation must be a finite scalar within [0, 1]")
        return info_saturation

    def _l11_noospheric_effect(self, l11_input: Dict[str, Any]) -> Dict[str, Any]:
        info_saturation = self._info_saturation(l11_input.get("info_saturation", 0.0))
        gaian_context = self._gaian_context(l11_input)
        structured_keys = {
            "boundary_shielding",
            "boundary_fragmentation_pressure",
            "polarization",
        }
        has_structured_diagnostics = any(key in l11_input for key in structured_keys)
        if not has_structured_diagnostics:
            stabilization_drive = (
                self.params.morphic_coupling * info_saturation * gaian_context["terminal_bandwidth"]
            )
            return {
                "entropy_load": 0.0,
                "stabilization_drive": stabilization_drive,
                "boundary_context_id": gaian_context["boundary_context_id"],
                "boundary_terminals": gaian_context["boundary_terminals"],
                "gaian_terminal_set": gaian_context["gaian_terminal_set"],
                "gaian_terminal_bandwidth": gaian_context["terminal_bandwidth"],
            }

        boundary_shielding = self._unit_scalar(
            l11_input.get("boundary_shielding", 0.0), "boundary_shielding"
        )
        fragmentation_pressure = self._nonnegative_scalar(
            l11_input.get("boundary_fragmentation_pressure", 0.0),
            "boundary_fragmentation_pressure",
        )
        polarization = self._nonnegative_scalar(l11_input.get("polarization", 0.0), "polarization")
        entropy_load = (
            info_saturation * (1.0 - boundary_shielding) + fragmentation_pressure + polarization
        )
        stabilization_drive = (
            self.params.morphic_coupling * info_saturation
            - self.params.noospheric_entropy_coupling * entropy_load
        )
        stabilization_drive *= gaian_context["terminal_bandwidth"]
        return {
            "entropy_load": float(entropy_load),
            "stabilization_drive": float(stabilization_drive),
            "boundary_context_id": gaian_context["boundary_context_id"],
            "boundary_terminals": gaian_context["boundary_terminals"],
            "gaian_terminal_set": gaian_context["gaian_terminal_set"],
            "gaian_terminal_bandwidth": gaian_context["terminal_bandwidth"],
        }

    @staticmethod
    def _gaian_context(l11_input: Dict[str, Any]) -> Dict[str, Any]:
        has_context_id = "boundary_context_id" in l11_input
        has_terminals = "boundary_terminals" in l11_input
        if not has_context_id and not has_terminals:
            return {
                "boundary_context_id": None,
                "boundary_terminals": (),
                "gaian_terminal_set": (),
                "terminal_bandwidth": 1.0,
            }
        if not has_context_id or not has_terminals:
            raise ValueError("boundary context requires boundary_context_id and boundary_terminals")

        raw_context_id = l11_input["boundary_context_id"]
        terminals = tuple(l11_input["boundary_terminals"])
        if raw_context_id is None and not terminals:
            return {
                "boundary_context_id": None,
                "boundary_terminals": (),
                "gaian_terminal_set": (),
                "terminal_bandwidth": 1.0,
            }
        context_id = str(raw_context_id)
        if not context_id:
            raise ValueError("boundary_context_id must be non-empty")
        valid_terminals = {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
        if not terminals or any(terminal not in valid_terminals for terminal in terminals):
            raise ValueError("boundary_terminals must contain valid T1-T7 terminal identifiers")

        gaian_terminals = tuple(
            terminal for terminal in terminals if terminal in {"T2", "T4", "T5"}
        )
        return {
            "boundary_context_id": context_id,
            "boundary_terminals": terminals,
            "gaian_terminal_set": gaian_terminals,
            "terminal_bandwidth": float(len(gaian_terminals) / 3.0),
        }

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
