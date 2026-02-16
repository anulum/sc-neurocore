"""
SCPN L7: Geometric-Symbolic Layer (Stochastic Implementation)
==============================================================

Implements Layer 7 of the SCPN framework: Geometric and symbolic
representations including sacred geometry, E8 lattice encoding,
acupuncture meridian mapping, and symbolic resonance patterns.

Key Features:
- Stochastic sacred geometry encoding (Phi, Metatron, Platonic)
- E8 lattice alignment metrics
- Symbolic health computation
- Acupuncture point activation patterns
- Integration with glyph streams

Author: Claude (Session 2026-01-31)
"""

from dataclasses import dataclass
import numpy as np
import logging
from typing import Dict, Optional

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

    def __init__(self, params: L7_StochasticParameters = None):
        self.params = params or L7_StochasticParameters()

        # Symbol activation states
        self.symbol_activations = np.random.random(self.params.n_symbols) * 0.3

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

        # E8 lattice representation (simplified 8D projection)
        self.e8_state = np.random.random(8) * 0.5

        # Time
        self.time = 0.0

    def step(
        self,
        dt: float,
        l6_input: Optional[Dict] = None,
        symbol_input: Optional[np.ndarray] = None,
        acupoint_stimulus: Optional[Dict[int, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            l6_input: Ecological layer output (Schumann, circadian).
            symbol_input: External symbolic input vector.
            acupoint_stimulus: Dict of {point_id: intensity} for acupuncture.

        Returns:
            Dict with glyph_vector, meridian_qi, sacred_geometry, output_bitstreams
        """
        self.time += dt

        # 1. Process symbol input
        if symbol_input is not None:
            self.symbol_activations = np.clip(
                self.symbol_activations + symbol_input[: self.params.n_symbols] * 0.2, 0.0, 1.0
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
        metatron_nodes = 13
        active_nodes = np.sum(self.symbol_activations[:metatron_nodes] > 0.5)
        self.metatron_flow = active_nodes / metatron_nodes
        # Add flow dynamics
        self.metatron_flow = 0.9 * self.metatron_flow + 0.1 * np.random.random()

        # 5. Compute Platonic solid coherence
        platonic_metrics = []
        for solid, vertices in self.PLATONIC_VERTICES.items():
            solid_activations = self.symbol_activations[:vertices]
            coherence = np.std(solid_activations)  # Lower std = more coherent
            platonic_metrics.append(1.0 - coherence)
        self.platonic_coherence = float(np.mean(platonic_metrics))

        # 6. E8 lattice alignment
        # Simplified: check alignment of 8D state vector with E8 root system
        # E8 has 240 roots; we use a proxy
        e8_norm = np.linalg.norm(self.e8_state)
        if e8_norm > 0:
            e8_unit = self.e8_state / e8_norm
            # Check alignment with simple E8 roots (permutations of ±1)
            simple_roots = np.eye(8)
            alignments = np.abs(np.dot(simple_roots, e8_unit))
            self.e8_alignment = float(np.max(alignments))
        else:
            self.e8_alignment = 0.5

        # Update E8 state with noise
        self.e8_state += 0.1 * np.random.normal(0, 1, 8) * dt
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

        # Ecological coupling (Schumann affects Qi)
        if l6_input is not None and "schumann_field" in l6_input:
            schumann_mean = np.mean(l6_input["schumann_field"])
            self.meridian_qi *= 0.9 + 0.1 * schumann_mean

        self.meridian_qi = np.clip(self.meridian_qi, 0.0, 1.0)

        # 9. Acupuncture point activation
        if acupoint_stimulus is not None:
            for point_id, intensity in acupoint_stimulus.items():
                if 0 <= point_id < self.params.n_acupoints:
                    self.acupoint_activations[point_id] = np.clip(
                        self.acupoint_activations[point_id] + intensity, 0.0, 1.0
                    )

        # Decay acupoint activations
        self.acupoint_activations *= 1.0 - self.params.symbol_decay * dt

        # 10. Assemble glyph vector
        self.glyph_vector = np.array(
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

        rands = np.random.random((self.params.n_symbols, self.params.bitstream_length))
        output_bitstreams = (rands < output_probs[:, None]).astype(np.uint8)

        return {
            "glyph_vector": self.glyph_vector.copy(),
            "phi_alignment": self.phi_alignment,
            "fibonacci_alignment": self.fibonacci_alignment,
            "metatron_flow": self.metatron_flow,
            "platonic_coherence": self.platonic_coherence,
            "e8_alignment": self.e8_alignment,
            "symbolic_health": self.symbolic_health,
            "meridian_qi": self.meridian_qi.copy(),
            "acupoint_activations": self.acupoint_activations.copy(),
            "e8_state": self.e8_state.copy(),
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        """Return the global symbolic coherence metric."""
        return self.symbolic_health

    def get_glyph_vector_normalized(self) -> np.ndarray:
        """Return normalized glyph vector for external use."""
        return self.glyph_vector / (np.max(self.glyph_vector) + 1e-8)

    def stimulate_meridian(self, meridian_id: int, intensity: float):
        """Stimulate a specific meridian."""
        if 0 <= meridian_id < self.params.n_meridians:
            self.meridian_qi[meridian_id] = np.clip(
                self.meridian_qi[meridian_id] + intensity, 0.0, 1.0
            )

    def get_acupoint_map(self) -> Dict[str, float]:
        """Return named acupoint activations (simplified set)."""
        # Classical acupoints (simplified)
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
