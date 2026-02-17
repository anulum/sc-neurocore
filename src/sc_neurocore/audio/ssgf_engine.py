"""
SSGF Engine -- Lightweight Stochastic Synthesis of Geometric Fields
=====================================================================

Pure-NumPy solver that couples Kuramoto phase oscillators with a
learned geometry matrix W(t), producing real-time audio-mapping
observables (binaural Hz, spatial angle, intensity, theurgic mode).

The architecture mirrors the full SSGF stack in SCPN-CODEBASE but is
self-contained: no JAX, no PyTorch, no ripser -- just numpy.

Author: Claude (Session 2026-02-17)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Canonical SCPN Natural Frequencies (16 layers) ───────────────────

OMEGA_N = np.array(
    [
        1.329,
        2.610,
        0.844,
        1.520,
        0.710,
        3.780,
        1.055,
        0.625,
        2.210,
        1.740,
        0.480,
        3.210,
        0.915,
        1.410,
        2.830,
        0.991,
    ]
)


def _build_knm(
    N: int = 16,
    K_base: float = 0.45,
    K_alpha: float = 0.3,
) -> np.ndarray:
    """Build N x N coupling matrix with exponential distance decay."""
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    K = K_base * np.exp(-K_alpha * dist)
    np.fill_diagonal(K, 0.0)

    # Calibration anchors
    anchors = [(0, 1, 0.302), (1, 2, 0.201), (2, 3, 0.252), (3, 4, 0.154)]
    for i, j, val in anchors:
        if i < N and j < N:
            K[i, j] = val
            K[j, i] = val

    # Cross-hierarchy boosts
    if N >= 16:
        K[0, 15] = max(K[0, 15], 0.05)
        K[15, 0] = max(K[15, 0], 0.05)
    if N >= 7:
        K[4, 6] = max(K[4, 6], 0.15)
        K[6, 4] = max(K[6, 4], 0.15)

    np.fill_diagonal(K, 0.0)
    return K


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class SSGFConfig:
    """All tuneable knobs for SSGFEngine."""

    N: int = 16
    z_dim: int = 120
    lr_z: float = 0.01
    sigma_g: float = 0.3
    micro_steps: int = 10
    dt: float = 0.001
    noise: float = 0.2
    K_base: float = 0.45
    K_alpha: float = 0.3
    field_pressure: float = 0.1
    seed: int = 42


# ── Engine ───────────────────────────────────────────────────────────


class SSGFEngine:
    """Lightweight SSGF geometry-coupled Kuramoto solver.

    Maintains a latent vector *z* whose decoded geometry matrix W(t)
    feeds back into the micro-cycle, steering oscillators toward
    higher global coherence R.  Audio-mapping observables are derived
    from the resulting phase dynamics and spectral properties of W.
    """

    def __init__(self, cfg: Optional[SSGFConfig] = None):
        self.cfg = cfg or SSGFConfig()
        c = self.cfg
        self._rng = np.random.RandomState(c.seed)

        # Phase state
        self.N = c.N
        self.omega = (
            OMEGA_N[: c.N].copy()
            if c.N <= 16
            else np.tile(
                OMEGA_N,
                (c.N // 16 + 1),
            )[: c.N].copy()
        )
        self.theta = self._rng.uniform(0, 2 * np.pi, c.N)

        # Coupling
        self.K = _build_knm(c.N, c.K_base, c.K_alpha)

        # Latent geometry
        self.z = self._rng.randn(c.z_dim).astype(np.float64) * 0.1
        self.W = self._decode(self.z)

        # Spectral cache
        self._eigvals = np.zeros(c.N)
        self._eigvecs = np.eye(c.N)

        # History for phase-velocity estimate
        self._prev_theta = self.theta.copy()

        # Running stats
        self.outer_step_count: int = 0
        self.R_global: float = 0.0
        self._cost_history: list[float] = []

    # ── Decoder: z -> W ──────────────────────────────────────────────

    def _decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector into a symmetric, non-negative weight
        matrix with zero diagonal via softplus on a symmetric shell."""
        N = self.N
        # Number of unique off-diagonal upper-triangle entries
        n_upper = N * (N - 1) // 2
        # Tile z to fill if z_dim < n_upper, or truncate
        flat = np.tile(z, (n_upper // len(z) + 1))[:n_upper]

        A = np.zeros((N, N))
        idx_upper = np.triu_indices(N, k=1)
        A[idx_upper] = flat
        A = A + A.T  # symmetric

        # Softplus: log(1 + exp(x)), numerically stable
        W = np.where(A > 20, A, np.log1p(np.exp(A)))
        np.fill_diagonal(W, 0.0)
        return W

    # ── Micro-Cycle ──────────────────────────────────────────────────

    def _micro_step(self) -> None:
        """One Kuramoto + geometry-feedback timestep (vectorised)."""
        c = self.cfg
        N = self.N
        theta = self.theta

        # Phase differences: diff[n, m] = theta[m] - theta[n]
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        sin_diff = np.sin(diff)

        # dtheta = omega + K coupling + geometry coupling + field + noise
        coupling_k = np.sum(self.K * sin_diff, axis=1)
        coupling_w = c.sigma_g * np.sum(self.W * sin_diff, axis=1)
        field_term = c.field_pressure * np.cos(theta)
        noise_term = c.noise * self._rng.randn(N)

        dtheta = self.omega + coupling_k + coupling_w + field_term + noise_term
        self.theta = (theta + dtheta * c.dt) % (2 * np.pi)

    # ── Spectral Bridge ──────────────────────────────────────────────

    def _spectral(self) -> None:
        """Compute eigendecomposition of the normalised Laplacian of W."""
        W = self.W
        d = W.sum(axis=1)
        d_safe = np.where(d > 1e-12, d, 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(d_safe)

        L_sym = np.eye(self.N) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
        # Force exact symmetry
        L_sym = 0.5 * (L_sym + L_sym.T)

        eigvals, eigvecs = np.linalg.eigh(L_sym)
        self._eigvals = eigvals
        self._eigvecs = eigvecs

    # ── Cost ─────────────────────────────────────────────────────────

    def _compute_R(self) -> float:
        """Kuramoto order parameter R = |<exp(i*theta)>|."""
        z_complex = np.mean(np.exp(1j * self.theta))
        return float(np.abs(z_complex))

    def _cost(self) -> float:
        """Composite cost: minimise negative coherence + regularise W."""
        R = self._compute_R()
        c_micro = 1.0 - R
        c_reg = 0.01 * np.sum(self.W**2) / (self.N * self.N)
        return c_micro + c_reg

    # ── Outer Cycle ──────────────────────────────────────────────────

    def outer_step(self) -> float:
        """One outer-cycle step: micro-cycle -> spectral -> grad update on z.

        Returns the cost after the step.
        """
        c = self.cfg

        # Save state
        self._prev_theta = self.theta.copy()

        # Run micro-cycle
        for _ in range(c.micro_steps):
            self._micro_step()

        # Spectral bridge
        self._spectral()

        # Update R
        self.R_global = self._compute_R()

        # Finite-difference gradient descent on z
        base_cost = self._cost()
        eps = 1e-4
        grad = np.zeros_like(self.z)

        for i in range(len(self.z)):
            z_plus = self.z.copy()
            z_plus[i] += eps
            W_backup = self.W
            self.W = self._decode(z_plus)
            cost_plus = self._cost()
            self.W = W_backup
            grad[i] = (cost_plus - base_cost) / eps

        self.z -= c.lr_z * grad
        self.W = self._decode(self.z)

        self.outer_step_count += 1
        self._cost_history.append(base_cost)
        return base_cost

    # ── Audio Mapping ────────────────────────────────────────────────

    def get_audio_mapping(self) -> Dict[str, float]:
        """Derive CCW audio parameters from current SSGF state.

        Returns
        -------
        dict with keys:
            binaural_hz      -- 0.5-40 Hz (from layer-2 phase velocity)
            pulse_rate        -- isochronic pulse rate (layer-4 coherence)
            spatial_angle     -- 0-360 degrees (layer-7 phase)
            intensity         -- 0-1 (from R_global)
            fiedler           -- algebraic connectivity of W
            spectral_gap      -- lambda_1 / lambda_2
            theurgic_mode     -- bool, True when R > 0.95
        """
        R = self.R_global

        # Layer 2 phase velocity -> binaural Hz (0.5 - 40)
        if self.N > 2:
            dphase_2 = (self.theta[1] - self._prev_theta[1]) / self.cfg.dt
            binaural_hz = float(np.clip(0.5 + abs(dphase_2) * 2.0, 0.5, 40.0))
        else:
            binaural_hz = 10.0

        # Layer 4 coherence -> pulse rate
        if self.N > 4:
            local_r = float(np.abs(np.mean(np.exp(1j * self.theta[3:5]))))
            pulse_rate = float(np.clip(2.0 + local_r * 18.0, 2.0, 20.0))
        else:
            pulse_rate = 8.0

        # Layer 7 phase -> spatial angle
        if self.N > 7:
            spatial_angle = float((self.theta[6] % (2 * np.pi)) / (2 * np.pi) * 360.0)
        else:
            spatial_angle = 0.0

        # R_global -> intensity
        intensity = float(np.clip(R, 0.0, 1.0))

        # Spectral properties
        fiedler = float(self._eigvals[1]) if len(self._eigvals) > 1 else 0.0
        spectral_gap = 0.0
        if len(self._eigvals) > 2 and abs(self._eigvals[2]) > 1e-12:
            spectral_gap = float(self._eigvals[1] / self._eigvals[2])

        theurgic = bool(R > 0.95)

        return {
            "binaural_hz": round(binaural_hz, 3),
            "pulse_rate": round(pulse_rate, 3),
            "spatial_angle": round(spatial_angle, 2),
            "intensity": round(intensity, 4),
            "fiedler": round(fiedler, 6),
            "spectral_gap": round(spectral_gap, 6),
            "theurgic_mode": theurgic,
        }

    # ── State ────────────────────────────────────────────────────────

    def get_state(self) -> Dict:
        """Full engine state snapshot."""
        return {
            "outer_step": self.outer_step_count,
            "R_global": round(self.R_global, 6),
            "theta": self.theta.tolist(),
            "z_norm": round(float(np.linalg.norm(self.z)), 6),
            "W_density": round(float(np.mean(self.W > 0.01)), 4),
            "W_mean": round(float(np.mean(self.W)), 6),
            "eigvals": [round(float(v), 6) for v in self._eigvals[:4]],
            "cost": round(self._cost_history[-1], 6) if self._cost_history else None,
            "audio": self.get_audio_mapping(),
        }
