# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L4 Cellular-Tissue Synchronization Layer

from typing import Any, Optional

"""
SCPN L4: Cellular-Tissue Synchronization Layer (Stochastic Implementation)
===========================================================================

Implements Layer 4 of the SCPN framework: Cellular and tissue-level
synchronization including gap junctions, calcium waves, and collective
oscillations.

Key Features:
- Stochastic cellular oscillator coupling
- Gap junction communication via bitstream propagation
- Calcium wave dynamics
- Tissue-level pattern formation

"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class L4_StochasticParameters:
    """Parameters for the Stochastic L4 Cellular Layer."""

    grid_size: Tuple[int, int] = (20, 20)  # 2D tissue grid
    bitstream_length: int = 1024

    # Oscillator parameters
    natural_frequency: float = 1.0  # Hz
    coupling_strength: float = 0.3
    noise_amplitude: float = 0.1

    # Gap junction dynamics
    gap_junction_conductance: float = 0.5
    gap_junction_noise: float = 0.05

    # Calcium dynamics
    ca_diffusion_rate: float = 0.1
    ca_decay_rate: float = 0.05
    ca_release_threshold: float = 0.6

    # Inter-layer coupling
    genomic_coupling: float = 0.1  # From L3
    organismal_coupling: float = 0.1  # To L5
    rng_seed: Optional[int] = None


class L4_CellularLayer:
    """
    Stochastic implementation of the Cellular-Tissue Synchronization Layer.

    Models collective cellular behavior, gap junction coupling, and
    tissue-level pattern formation using bitstream representations.
    """

    def __init__(self, params: Optional[L4_StochasticParameters] = None):
        self.params = params or L4_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)
        self.n_cells = self.params.grid_size[0] * self.params.grid_size[1]

        # Oscillator phases (Kuramoto-like model)
        self.phases = self._rng.random(self.n_cells) * 2 * np.pi

        # Oscillator amplitudes
        self.amplitudes = np.ones(self.n_cells) * 0.5

        # Calcium concentrations
        self.calcium = self._rng.random(self.n_cells) * 0.3

        # Gap junction states (0 = closed, 1 = open)
        self.gap_junctions = self._init_gap_junctions()

        # Tissue activity pattern
        self.activity_pattern = np.zeros(self.n_cells)

        # Build neighbor connectivity matrix
        self.neighbors = self._build_neighbor_matrix()

    def _init_gap_junctions(self) -> np.ndarray[Any, Any]:
        """Initialize gap junction connectivity."""
        # Random initial state with bias toward open
        return (self._rng.random(self.n_cells) > 0.3).astype(np.float32)

    def _build_neighbor_matrix(self) -> np.ndarray[Any, Any]:
        """Build 2D grid neighbor connectivity matrix."""
        h, w = self.params.grid_size
        n = self.n_cells
        neighbors = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            row, col = i // w, i % w
            # 4-connectivity (von Neumann)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w:
                    j = nr * w + nc
                    neighbors[i, j] = 1.0

        return neighbors

    def step(
        self,
        dt: float,
        l3_input: Optional[Dict[str, Any]] = None,
        external_stimulus: Optional[np.ndarray[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            l3_input: Genomic layer output (protein levels).
            external_stimulus: External stimulation pattern.

        Returns:
            Dict with phases, calcium, synchronization, output_bitstreams
        """
        self._validate_step_inputs(dt, l3_input, external_stimulus, self.n_cells)
        # 1. Kuramoto oscillator dynamics
        # dθ/dt = ω + K/N * Σ sin(θ_j - θ_i)
        phase_diffs = np.sin(self.phases[None, :] - self.phases[:, None])
        coupling_term = (
            self.params.coupling_strength
            * np.sum(self.neighbors * phase_diffs, axis=1)
            / np.maximum(np.sum(self.neighbors, axis=1), 1)
        )

        noise = self.params.noise_amplitude * self._rng.normal(0, 1, self.n_cells)

        self.phases += (2 * np.pi * self.params.natural_frequency + coupling_term + noise) * dt
        self.phases = self.phases % (2 * np.pi)

        # 2. Calcium wave dynamics
        # Diffusion via gap junctions
        ca_diff = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            neighbor_indices = np.where(self.neighbors[i] > 0)[0]
            if len(neighbor_indices) > 0:
                # Diffusion weighted by gap junction state
                for j in neighbor_indices:
                    gj_state = (self.gap_junctions[i] + self.gap_junctions[j]) / 2
                    ca_diff[i] += (
                        self.params.gap_junction_conductance
                        * gj_state
                        * (self.calcium[j] - self.calcium[i])
                    )

        self.calcium += (
            self.params.ca_diffusion_rate * ca_diff - self.params.ca_decay_rate * self.calcium
        ) * dt

        # Calcium-induced calcium release (CICR)
        cicr_mask = self.calcium > self.params.ca_release_threshold
        self.calcium = np.where(cicr_mask, self.calcium + 0.2, self.calcium)
        self.calcium = np.clip(self.calcium, 0.0, 1.0)

        # 3. Gap junction dynamics
        # Gap junctions modulated by calcium and coupling
        gj_noise = self.params.gap_junction_noise * self._rng.normal(0, 1, self.n_cells)
        self.gap_junctions = np.clip(
            self.gap_junctions + gj_noise * dt + 0.1 * (1 - self.calcium) * dt, 0.0, 1.0
        )

        # 4. Genomic input coupling (L3 proteins modulate oscillators)
        if l3_input is not None and "protein_levels" in l3_input:
            protein_mean = self._finite_mean(l3_input["protein_levels"], "protein_levels")
            self.amplitudes = np.clip(
                self.amplitudes + protein_mean * self.params.genomic_coupling * dt, 0.1, 1.0
            )

        # 5. External stimulus
        if external_stimulus is not None:
            self.calcium += external_stimulus[: self.n_cells] * dt
            self.calcium = np.clip(self.calcium, 0.0, 1.0)

        # 6. Compute activity pattern
        self.activity_pattern = self.amplitudes * (1 + np.cos(self.phases)) / 2

        # 7. Compute synchronization order parameter
        order_parameter = float(np.abs(np.mean(np.exp(1j * self.phases))))

        # 8. Generate output bitstreams
        output_probs = np.clip(self.activity_pattern, 0.0, 1.0)
        rands = self._rng.random((self.n_cells, self.params.bitstream_length))
        output_bitstreams = (rands < output_probs[:, None]).astype(np.uint8)
        organismal_drive = self.params.organismal_coupling * order_parameter

        return {
            "phases": self.phases.copy(),
            "amplitudes": self.amplitudes.copy(),
            "calcium": self.calcium.copy(),
            "gap_junctions": self.gap_junctions.copy(),
            "activity_pattern": self.activity_pattern.copy(),
            "synchronization": order_parameter,
            "organismal_drive": organismal_drive,
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        """Return the global synchronization metric (Kuramoto order parameter)."""
        return float(np.abs(np.mean(np.exp(1j * self.phases))))

    def get_tissue_pattern(self) -> np.ndarray[Any, Any]:
        """Return 2D tissue activity pattern."""
        return self.activity_pattern.reshape(self.params.grid_size)

    @staticmethod
    def _validate_params(params: L4_StochasticParameters) -> None:
        if (
            not isinstance(params.grid_size, tuple)
            or len(params.grid_size) != 2
            or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                for dim in params.grid_size
            )
        ):
            raise ValueError("grid_size must be a tuple of two positive integers")
        if (
            not isinstance(params.bitstream_length, int)
            or isinstance(params.bitstream_length, bool)
            or params.bitstream_length <= 0
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if not math.isfinite(float(params.natural_frequency)) or params.natural_frequency <= 0.0:
            raise ValueError("natural_frequency must be finite and positive")
        for field_name in (
            "coupling_strength",
            "noise_amplitude",
            "gap_junction_noise",
            "ca_diffusion_rate",
            "ca_decay_rate",
            "genomic_coupling",
            "organismal_coupling",
        ):
            value = float(getattr(params, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if (
            not math.isfinite(float(params.gap_junction_conductance))
            or params.gap_junction_conductance < 0.0
            or params.gap_junction_conductance > 1.0
        ):
            raise ValueError("gap_junction_conductance must be finite and within [0, 1]")
        if (
            not math.isfinite(float(params.ca_release_threshold))
            or params.ca_release_threshold < 0.0
            or params.ca_release_threshold > 1.0
        ):
            raise ValueError("ca_release_threshold must be finite and within [0, 1]")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @classmethod
    def _validate_step_inputs(
        cls,
        dt: float,
        l3_input: Optional[Dict[str, Any]],
        external_stimulus: Optional[np.ndarray[Any, Any]],
        n_cells: int,
    ) -> None:
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if l3_input is not None and "protein_levels" in l3_input:
            cls._finite_mean(l3_input["protein_levels"], "protein_levels")
        if external_stimulus is not None:
            stimulus = np.asarray(external_stimulus, dtype=np.float64)
            if stimulus.size != n_cells or not np.all(np.isfinite(stimulus)):
                raise ValueError("external_stimulus must contain one finite value per cell")

    @staticmethod
    def _finite_mean(values: Any, name: str) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite values")
        return float(np.mean(arr))
