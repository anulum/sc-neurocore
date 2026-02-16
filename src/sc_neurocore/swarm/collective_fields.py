"""
Collective Fields — SCPN-inspired shared substrate
====================================================

Implements three shared fields inspired by SCPN layers:

- Chemical field (L2): Scalar diffusion on a grid, agents secrete
  attractant based on neural activity. Laplacian diffusion + decay.

- Emotional field (L5): Per-agent 8-dim emotion vector. Mean-field
  coupling pulls emotions toward swarm mean (collective mood).

- Symbolic field (L7): 2D grid of 2-component glyph vectors. Agents
  imprint symbolic patterns; nearest-neighbor resonance.

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FieldConfig:
    """Configuration for collective fields."""
    grid_resolution: int = 50  # Grid cells per dimension
    arena_width: float = 100.0
    arena_height: float = 100.0
    # Chemical field (L2)
    chem_diffusion_rate: float = 0.1
    chem_decay_rate: float = 0.02
    chem_secretion_scale: float = 5.0
    # Emotional field (L5)
    n_emotional_dims: int = 8
    emotion_coupling: float = 0.1
    emotion_decay: float = 0.01
    # Symbolic field (L7)
    symbolic_dims: int = 2
    symbolic_diffusion: float = 0.05
    symbolic_decay: float = 0.01


class CollectiveFields:
    """
    Three SCPN-inspired shared fields for inter-agent communication.
    """

    def __init__(self, config: Optional[FieldConfig] = None):
        self.config = config or FieldConfig()
        c = self.config
        g = c.grid_resolution

        # L2: Chemical concentration grid
        self.chemical_field = np.zeros((g, g), dtype=np.float64)

        # L5: Per-agent emotional state (set externally)
        # This is managed per-agent but synchronized here
        self._emotion_mean = np.full(c.n_emotional_dims, 0.5)

        # L7: Symbolic glyph grid
        self.symbolic_field = np.zeros((g, g, c.symbolic_dims), dtype=np.float64)

        # Precompute Laplacian kernel for diffusion
        self._lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    def _pos_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world position to grid cell."""
        g = self.config.grid_resolution
        gx = int(np.clip(x / self.config.arena_width * g, 0, g - 1))
        gy = int(np.clip(y / self.config.arena_height * g, 0, g - 1))
        return gx, gy

    # ── Chemical Field (L2) ──────────────────────────────────────────

    def deposit_chemical(self, x: float, y: float, amount: float):
        """Agent deposits chemical at its position."""
        gx, gy = self._pos_to_grid(x, y)
        self.chemical_field[gx, gy] += amount * self.config.chem_secretion_scale

    def get_chemical_gradient(self, x: float, y: float) -> np.ndarray:
        """Get chemical gradient (dx, dy) at a position."""
        gx, gy = self._pos_to_grid(x, y)
        g = self.config.grid_resolution

        # Finite differences
        dx = 0.0
        dy = 0.0
        if gx > 0 and gx < g - 1:
            dx = (self.chemical_field[gx + 1, gy] - self.chemical_field[gx - 1, gy]) / 2.0
        if gy > 0 and gy < g - 1:
            dy = (self.chemical_field[gx, gy + 1] - self.chemical_field[gx, gy - 1]) / 2.0

        # Normalize to [0, 1]
        mag = np.sqrt(dx ** 2 + dy ** 2)
        if mag > 0:
            dx /= mag
            dy /= mag
        return np.array([(dx + 1) / 2, (dy + 1) / 2])  # Map [-1,1] to [0,1]

    def diffuse_chemical(self, dt: float = 1.0):
        """Apply Laplacian diffusion and decay to chemical field."""
        from scipy.ndimage import convolve
        lap = convolve(self.chemical_field, self._lap_kernel, mode="constant", cval=0.0)
        self.chemical_field += self.config.chem_diffusion_rate * lap * dt
        self.chemical_field *= (1.0 - self.config.chem_decay_rate * dt)
        self.chemical_field = np.clip(self.chemical_field, 0, 10.0)

    # ── Emotional Field (L5) ─────────────────────────────────────────

    def synchronize_emotions(
        self, emotions: List[np.ndarray], coupling: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Pull each agent's emotions toward swarm mean.

        Args:
            emotions: List of (n_emotional_dims,) arrays, one per agent.
            coupling: Override coupling strength.

        Returns:
            Updated emotion vectors.
        """
        if not emotions:
            return emotions

        c = coupling if coupling is not None else self.config.emotion_coupling
        emotion_stack = np.array(emotions)
        self._emotion_mean = emotion_stack.mean(axis=0)

        updated = []
        for emo in emotions:
            new_emo = emo + c * (self._emotion_mean - emo)
            # Decay toward neutral (0.5)
            new_emo += self.config.emotion_decay * (0.5 - new_emo)
            updated.append(np.clip(new_emo, 0, 1))
        return updated

    def get_emotion_mean(self) -> np.ndarray:
        """Return current swarm emotional mean."""
        return self._emotion_mean.copy()

    # ── Symbolic Field (L7) ──────────────────────────────────────────

    def deposit_symbolic(self, x: float, y: float, glyph: np.ndarray):
        """Agent imprints a symbolic glyph at its position."""
        gx, gy = self._pos_to_grid(x, y)
        self.symbolic_field[gx, gy] += glyph[:self.config.symbolic_dims]

    def get_symbolic_value(self, x: float, y: float) -> np.ndarray:
        """Get symbolic field value at position."""
        gx, gy = self._pos_to_grid(x, y)
        val = self.symbolic_field[gx, gy].copy()
        # Normalize to [0, 1]
        mag = np.linalg.norm(val)
        if mag > 1.0:
            val /= mag
        return np.clip((val + 1) / 2, 0, 1)  # Map to [0, 1]

    def diffuse_symbolic(self, dt: float = 1.0):
        """Apply diffusion and decay to symbolic field."""
        from scipy.ndimage import convolve
        for d in range(self.config.symbolic_dims):
            lap = convolve(
                self.symbolic_field[:, :, d],
                self._lap_kernel,
                mode="constant",
                cval=0.0,
            )
            self.symbolic_field[:, :, d] += self.config.symbolic_diffusion * lap * dt
            self.symbolic_field[:, :, d] *= (1.0 - self.config.symbolic_decay * dt)

    # ── Combined Update ──────────────────────────────────────────────

    def step(self, dt: float = 1.0):
        """Diffuse all fields."""
        self.diffuse_chemical(dt)
        self.diffuse_symbolic(dt)

    def reset(self):
        """Reset all fields to zero."""
        g = self.config.grid_resolution
        self.chemical_field = np.zeros((g, g))
        self.symbolic_field = np.zeros((g, g, self.config.symbolic_dims))
        self._emotion_mean = np.full(self.config.n_emotional_dims, 0.5)

    def get_state(self) -> dict:
        """Return field state for visualization."""
        return {
            "chemical_field_sum": float(self.chemical_field.sum()),
            "chemical_field_max": float(self.chemical_field.max()),
            "symbolic_field_norm": float(np.linalg.norm(self.symbolic_field)),
            "emotion_mean": self._emotion_mean.tolist(),
        }
