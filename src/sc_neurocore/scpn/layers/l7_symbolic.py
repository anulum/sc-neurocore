# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L7 Geometric-Symbolic Layer (Stochastic Implementation)

"""SCPN L7: geometric-symbolic layer (stochastic implementation).

Implements Layer 7 of the SCPN framework: Geometric and symbolic
representations including sacred geometry, E8 lattice encoding,
acupuncture meridian mapping, and symbolic resonance patterns.

Key features: stochastic sacred-geometry encoding (Phi, Metatron, Platonic),
E8 lattice alignment metrics, symbolic health computation, acupuncture point
activation patterns, and integration with glyph streams.
"""

from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class L7_StochasticParameters:
    """Parameters for the Stochastic L7 Symbolic Layer."""

    n_symbols: int = 128
    n_meridians: int = 12  # TCM meridians
    n_acupoints: int = 361  # Classical acupuncture points
    bitstream_length: int = 1024

    # Sacred geometry parameters
    phi_alignment_weight: float = 0.25
    fibonacci_weight: float = 0.2
    metatron_weight: float = 0.2
    platonic_weight: float = 0.15
    e8_weight: float = 0.2

    # Symbolic dynamics
    symbol_decay: float = 0.05
    symbol_coupling: float = 0.3

    # Glyph stream parameters
    glyph_dimensions: int = 6  # Phi, Fib, Metatron, Platonic, E8, Health

    # Inter-layer coupling
    ecological_coupling: float = 0.1  # From L6
    cosmic_coupling: float = 0.15  # To L8
    rng_seed: Optional[int] = None


class L7_SymbolicLayer:
    """
    Stochastic implementation of the Geometric-Symbolic Layer.

    Models sacred geometry patterns, symbolic resonances, and
    acupuncture point dynamics using bitstream representations.
    """

    # Platonic solid vertex counts
    PLATONIC_VERTICES = {
        "tetrahedron": 4,
        "cube": 8,
        "octahedron": 6,
        "dodecahedron": 20,
        "icosahedron": 12,
    }

    # Fibonacci sequence for alignment
    FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    def __init__(self, params: Optional[L7_StochasticParameters] = None):
        self.params = params or L7_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)

        # Symbol activation states
        self.symbol_activations = self._rng.random(self.params.n_symbols) * 0.3

        # Sacred geometry metrics
        self.phi_alignment = 0.5
        self.fibonacci_alignment = 0.5
        self.metatron_flow = 0.5
        self.platonic_coherence = 0.5
        self.e8_alignment = 0.5
        self.symbolic_health = 0.5

        # Meridian states (TCM)
        self.meridian_qi = np.ones(self.params.n_meridians) * 0.5

        # Acupuncture point activations
        self.acupoint_activations = np.zeros(self.params.n_acupoints)

        # Glyph vector (normalized output)
        self.glyph_vector = np.zeros(self.params.glyph_dimensions)

        # E8 lattice representation in the canonical 8D root-space basis.
        self.e8_state = self._rng.random(8) * 0.5

        # Time
        self.time = 0.0

    def step(
        self,
        dt: float,
        l6_input: Optional[Dict[str, Any]] = None,
        symbol_input: Optional[np.ndarray[Any, Any]] = None,
        acupoint_stimulus: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            l6_input: Ecological layer output (Schumann, circadian).
            symbol_input: External symbolic input vector.
            acupoint_stimulus: Dict of {point_id: intensity} for acupuncture.

        Returns
        -------
            Dict with glyph_vector, meridian_qi, sacred_geometry, output_bitstreams
        """
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time += dt

        # 1. Process symbol input
        if symbol_input is not None:
            symbol_values = self._symbol_input(symbol_input)
            self.symbol_activations = np.clip(
                self.symbol_activations + symbol_values * 0.2, 0.0, 1.0
            )

        # 2. Compute Phi (Golden Ratio) alignment
        # Check how close symbol ratios are to Phi
        sorted_activations = np.sort(self.symbol_activations)[::-1]
        if sorted_activations[1] > 0.01:
            ratios = sorted_activations[:-1] / (sorted_activations[1:] + 1e-8)
            phi_distances = np.abs(ratios - PHI)
            self.phi_alignment = float(np.exp(-np.mean(phi_distances)))
        else:
            self.phi_alignment = 0.5

        # 3. Compute Fibonacci alignment
        # Check if activation levels follow Fibonacci ratios
        fib_normalized = np.array(self.FIBONACCI[:8]) / self.FIBONACCI[7]
        top_8 = sorted_activations[:8]
        if np.max(top_8) > 0.01:
            top_8_norm = top_8 / (np.max(top_8) + 1e-8)
            fib_corr = np.corrcoef(top_8_norm, fib_normalized)[0, 1]
            self.fibonacci_alignment = float(max(0, (fib_corr + 1) / 2))
        else:
            self.fibonacci_alignment = 0.5

        # 4. Compute Metatron's Cube flow
        # Based on 13-sphere / 78-line connectivity pattern
        metatron_nodes = min(13, self.params.n_symbols)
        active_nodes = np.sum(self.symbol_activations[:metatron_nodes] > 0.5)
        self.metatron_flow = active_nodes / metatron_nodes
        # Add flow dynamics
        self.metatron_flow = 0.9 * self.metatron_flow + 0.1 * self._rng.random()

        # 5. Compute Platonic solid coherence
        platonic_metrics = []
        for solid, vertices in self.PLATONIC_VERTICES.items():
            solid_activations = self.symbol_activations[:vertices]
            coherence = np.std(solid_activations)  # Lower std = more coherent
            platonic_metrics.append(1.0 - coherence)
        self.platonic_coherence = float(np.mean(platonic_metrics))

        # 6. E8 lattice alignment against the full 240-root system.
        e8_norm = np.linalg.norm(self.e8_state)
        if e8_norm > 0:
            e8_unit = self.e8_state / e8_norm
            roots = self._e8_roots()
            root_units = roots / np.linalg.norm(roots, axis=1, keepdims=True)
            alignments = np.abs(root_units @ e8_unit)
            self.e8_alignment = float(np.max(alignments))
        else:
            self.e8_alignment = 0.5

        # Update E8 state with noise
        self.e8_state += 0.1 * self._rng.normal(0.0, 1.0, 8) * dt
        self.e8_state = np.clip(self.e8_state, -1, 1)

        # 7. Compute symbolic health
        self.symbolic_health = (
            self.params.phi_alignment_weight * self.phi_alignment
            + self.params.fibonacci_weight * self.fibonacci_alignment
            + self.params.metatron_weight * self.metatron_flow
            + self.params.platonic_weight * self.platonic_coherence
            + self.params.e8_weight * self.e8_alignment
        )

        # 8. Meridian Qi dynamics
        # Qi flows through meridians with circadian modulation
        qi_flow = np.roll(self.meridian_qi, 1) - self.meridian_qi
        self.meridian_qi += qi_flow * self.params.symbol_coupling * dt

        # Ecological coupling (structured L6 symbolic drive, with Schumann fallback)
        if l6_input is not None:
            l6_effect = self._l6_symbolic_effect(l6_input)
            if l6_effect != 0.0:
                self.meridian_qi *= 1.0 + self.params.ecological_coupling * l6_effect

        self.meridian_qi = np.clip(self.meridian_qi, 0.0, 1.0)

        # 9. Acupuncture point activation
        if acupoint_stimulus is not None:
            for point_id, intensity in self._acupoint_stimulus(acupoint_stimulus).items():
                self.acupoint_activations[point_id] = np.clip(
                    self.acupoint_activations[point_id] + intensity, 0.0, 1.0
                )

        # Decay acupoint activations
        self.acupoint_activations *= 1.0 - self.params.symbol_decay * dt

        # 10. Assemble glyph vector
        self.glyph_vector[...] = np.array(
            [
                self.phi_alignment,
                self.fibonacci_alignment,
                self.metatron_flow,
                self.platonic_coherence,
                self.e8_alignment,
                self.symbolic_health,
            ]
        )

        # 11. Symbol dynamics (decay and coupling)
        # Coupling: nearby symbols influence each other
        coupling = np.roll(self.symbol_activations, 1) + np.roll(self.symbol_activations, -1)
        self.symbol_activations += (
            self.params.symbol_coupling * (coupling / 2 - self.symbol_activations) * dt
        )
        # Decay
        self.symbol_activations *= 1.0 - self.params.symbol_decay * dt
        self.symbol_activations = np.clip(self.symbol_activations, 0.0, 1.0)

        # 12. Generate output bitstreams
        output_probs = np.concatenate(
            [self.symbol_activations, self.meridian_qi, self.glyph_vector]
        )
        output_probs = output_probs[: self.params.n_symbols]

        rands = self._rng.random((self.params.n_symbols, self.params.bitstream_length))
        output_bitstreams = (rands < output_probs[:, None]).astype(np.uint8)

        return {
            "glyph_vector": self.glyph_vector.copy(),
            "phi_alignment": self.phi_alignment,
            "fibonacci_alignment": self.fibonacci_alignment,
            "metatron_flow": self.metatron_flow,
            "platonic_coherence": self.platonic_coherence,
            "e8_alignment": self.e8_alignment,
            "symbolic_health": self.symbolic_health,
            "cosmic_phase_drive": self.params.cosmic_coupling * self.symbolic_health,
            "meridian_qi": self.meridian_qi.copy(),
            "acupoint_activations": self.acupoint_activations.copy(),
            "e8_state": self.e8_state.copy(),
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        """Return the global symbolic coherence metric."""
        return self.symbolic_health

    def get_glyph_vector_normalized(self) -> np.ndarray[Any, Any]:
        """Return normalized glyph vector for external use."""
        return self.glyph_vector / (np.max(self.glyph_vector) + 1e-8)

    def stimulate_meridian(self, meridian_id: int, intensity: float) -> None:
        """Stimulate a specific meridian."""
        if not 0 <= meridian_id < self.params.n_meridians:
            raise ValueError("meridian_id must be in range")
        if not math.isfinite(float(intensity)):
            raise ValueError("intensity must be finite")
        self.meridian_qi[meridian_id] = np.clip(self.meridian_qi[meridian_id] + intensity, 0.0, 1.0)

    def get_acupoint_map(self) -> Dict[str, float]:
        """Return clinically common named acupoint activations."""
        named_points = {
            "LI4_Hegu": 4,
            "ST36_Zusanli": 36,
            "SP6_Sanyinjiao": 60,
            "PC6_Neiguan": 96,
            "LV3_Taichong": 120,
            "GV20_Baihui": 200,
            "CV4_Guanyuan": 250,
            "BL23_Shenshu": 300,
        }
        return {
            name: float(self.acupoint_activations[idx])
            for name, idx in named_points.items()
            if idx < self.params.n_acupoints
        }

    @staticmethod
    def _e8_roots() -> np.ndarray[Any, Any]:
        roots: list[np.ndarray[Any, Any]] = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        root = np.zeros(8, dtype=np.float64)
                        root[i] = si
                        root[j] = sj
                        roots.append(root)
        for mask in range(256):
            signs = np.array(
                [-1.0 if (mask >> bit) & 1 else 1.0 for bit in range(8)],
                dtype=np.float64,
            )
            if np.count_nonzero(signs < 0.0) % 2 == 0:
                roots.append(0.5 * signs)
        return np.asarray(roots, dtype=np.float64)

    @staticmethod
    def _validate_params(params: L7_StochasticParameters) -> None:
        if not isinstance(params.n_symbols, int) or isinstance(params.n_symbols, bool):
            raise ValueError("n_symbols must be a positive integer")
        if params.n_symbols < 2:
            raise ValueError("n_symbols must be at least two")
        if not isinstance(params.n_meridians, int) or isinstance(params.n_meridians, bool):
            raise ValueError("n_meridians must be a positive integer")
        if params.n_meridians <= 0:
            raise ValueError("n_meridians must be positive")
        if not isinstance(params.n_acupoints, int) or isinstance(params.n_acupoints, bool):
            raise ValueError("n_acupoints must be a positive integer")
        if params.n_acupoints <= 0:
            raise ValueError("n_acupoints must be positive")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if params.glyph_dimensions != 6:
            raise ValueError("glyph_dimensions must be six for the current glyph contract")
        weights = np.array(
            [
                params.phi_alignment_weight,
                params.fibonacci_weight,
                params.metatron_weight,
                params.platonic_weight,
                params.e8_weight,
            ],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("weights must be finite, non-negative, and sum positive")
        for field_name in (
            "symbol_decay",
            "symbol_coupling",
            "ecological_coupling",
            "cosmic_coupling",
        ):
            value = float(getattr(params, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    def _symbol_input(self, symbol_input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        values = np.asarray(symbol_input, dtype=np.float64).reshape(-1)
        if values.size < self.params.n_symbols:
            raise ValueError("symbol_input must contain at least n_symbols values")
        trimmed = values[: self.params.n_symbols]
        if not np.all(np.isfinite(trimmed)):
            raise ValueError("symbol_input must contain only finite values")
        return trimmed

    @staticmethod
    def _finite_mean(value: Any, name: str) -> float:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError(f"{name} must contain at least one value")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return float(np.mean(values))

    @classmethod
    def _l6_symbolic_effect(cls, l6_input: Dict[str, Any]) -> float:
        if "symbolic_drive" in l6_input:
            return cls._unit_mean(l6_input["symbolic_drive"], "symbolic_drive")
        if "schumann_field" in l6_input:
            return cls._finite_mean(l6_input["schumann_field"], "schumann_field") - 1.0
        return 0.0

    @staticmethod
    def _unit_mean(value: Any, name: str) -> float:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError(f"{name} must contain at least one value")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{name} values must be within [0, 1]")
        return float(np.mean(values))

    def _acupoint_stimulus(self, stimulus: Dict[int, float]) -> Dict[int, float]:
        validated: Dict[int, float] = {}
        for point_id, intensity in stimulus.items():
            if not isinstance(point_id, int) or isinstance(point_id, bool):
                raise ValueError("acupoint_stimulus keys must be integer point ids")
            if not 0 <= point_id < self.params.n_acupoints:
                raise ValueError("acupoint_stimulus point id out of range")
            intensity_value = float(intensity)
            if not math.isfinite(intensity_value):
                raise ValueError("acupoint_stimulus intensities must be finite")
            validated[point_id] = intensity_value
        return validated
