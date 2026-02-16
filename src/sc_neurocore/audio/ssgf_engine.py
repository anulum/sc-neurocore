"""
SSGF Geometry Engine (Lightweight)
===================================

Implements the core SSGF cycle for adaptive audio:
1. Latent vector z → geometry matrix W via softplus decoder
2. W → Kuramoto micro-cycles with geometry feedback
3. W → spectral decomposition (eigenvalues for audio mapping)
4. Cost computation → gradient update on z

This is a self-contained version optimized for real-time audio adaptation.

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SSGFConfig:
    """SSGF engine configuration."""
    N: int = 16                     # Number of oscillators
    latent_dim: int = 120           # N*(N-1)/2 for upper triangle
    lr_z: float = 0.01             # Latent learning rate
    sigma_g: float = 0.3           # Geometry feedback strength
    micro_steps: int = 10          # Kuramoto steps per outer step
    dt: float = 0.01               # Micro timestep
    noise_std: float = 0.05        # Kuramoto noise
    K_base: float = 0.45           # Base coupling
    alpha_decay: float = 0.3       # Distance decay
    # Cost weights
    w_micro: float = 1.0
    w_spectral: float = 0.5
    w_reg: float = 0.1
    # Field pressure for SSGF gradient
    field_pressure: float = 0.1
    seed: int = 42


@dataclass
class SSGFState:
    """Current SSGF state snapshot."""
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    W: np.ndarray = field(default_factory=lambda: np.array([]))
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    eigvals: np.ndarray = field(default_factory=lambda: np.array([]))
    R_global: float = 0.0
    fiedler: float = 0.0
    spectral_gap: float = 0.0
    C_micro: float = 0.0
    U_total: float = 0.0
    outer_step: int = 0

    def to_dict(self) -> Dict:
        return {
            "R_global": round(self.R_global, 4),
            "fiedler": round(self.fiedler, 4),
            "spectral_gap": round(self.spectral_gap, 4),
            "C_micro": round(self.C_micro, 4),
            "U_total": round(self.U_total, 4),
            "outer_step": self.outer_step,
        }


class SSGFEngine:
    """
    Lightweight SSGF geometry engine for adaptive audio.

    Maintains latent vector z that decodes to geometry W.
    Each outer step runs micro-cycles (Kuramoto + geometry feedback),
    computes costs, and updates z via finite-difference gradient descent.
    """

    def __init__(self, config: Optional[SSGFConfig] = None):
        self.config = config or SSGFConfig()
        c = self.config
        self.rng = np.random.RandomState(c.seed)

        # Canonical Kuramoto parameters
        self.omega = np.array(
            [1.329, 1.261, 1.198, 1.140, 1.085, 1.034, 0.987, 1.044,
             1.106, 1.172, 1.015, 0.967, 1.023, 1.083, 1.147, 0.991],
        )[:c.N]

        # Build base coupling K
        self.K = np.zeros((c.N, c.N))
        for i in range(c.N):
            for j in range(c.N):
                if i != j:
                    self.K[i, j] = c.K_base * np.exp(-c.alpha_decay * abs(i - j))

        # Latent vector z
        self.z = self.rng.normal(0, 0.1, c.latent_dim)

        # Oscillator phases
        self.theta = self.rng.uniform(0, 2 * np.pi, c.N)

        # Decode initial W
        self.W = self._decode(self.z)
        self.eigvals = np.zeros(c.N)
        self._outer_step = 0

    def _decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent z → symmetric non-negative geometry W."""
        N = self.config.N
        # Build symmetric matrix from upper triangle
        A = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if idx < len(z):
                    A[i, j] = z[idx]
                    A[j, i] = z[idx]
                    idx += 1
        # Softplus ensures non-negative
        W = np.log1p(np.exp(A))
        np.fill_diagonal(W, 0.0)
        return W

    def _micro_step(self, theta: np.ndarray, W: np.ndarray) -> np.ndarray:
        """One Kuramoto micro-step with geometry feedback."""
        c = self.config
        N = c.N
        coupling = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    sin_diff = np.sin(theta[j] - theta[i])
                    coupling[i] += (self.K[i, j] + c.sigma_g * W[i, j]) * sin_diff

        noise = self.rng.normal(0, c.noise_std, N)
        theta_new = theta + (self.omega + coupling + noise) * c.dt
        return theta_new % (2 * np.pi)

    def _compute_R(self, theta: np.ndarray) -> float:
        """Kuramoto order parameter."""
        z = np.exp(1j * theta)
        return float(np.abs(z.mean()))

    def _spectral(self, W: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Compute spectral properties of W."""
        N = W.shape[0]
        D = np.diag(W.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-8)))
        L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
        eigvals = np.sort(np.linalg.eigvalsh(L_sym))
        fiedler = float(eigvals[1]) if N > 1 else 0.0
        spectral_gap = float(eigvals[1] / max(eigvals[2], 1e-8)) if N > 2 else 0.0
        return eigvals, fiedler, spectral_gap

    def _compute_costs(self, theta: np.ndarray, W: np.ndarray) -> Dict[str, float]:
        """Compute SSGF cost terms."""
        R = self._compute_R(theta)
        C_micro = 1.0 - R  # Want R → 1
        C_spectral = max(0, 0.1 - self._spectral(W)[1])  # Want fiedler > 0.1
        C_reg = float(np.sum(W ** 2)) / (W.shape[0] ** 2)  # Frobenius regularization
        c = self.config
        U_total = c.w_micro * C_micro + c.w_spectral * C_spectral + c.w_reg * C_reg
        return {
            "C_micro": C_micro,
            "C_spectral": C_spectral,
            "C_reg": C_reg,
            "U_total": U_total,
            "R_global": R,
        }

    def outer_step(self) -> SSGFState:
        """
        One outer SSGF cycle:
        1. Run micro-cycles with current W
        2. Compute costs
        3. Update z via finite-difference gradient
        """
        c = self.config

        # 1. Micro-cycles
        for _ in range(c.micro_steps):
            self.theta = self._micro_step(self.theta, self.W)

        # 2. Costs at current z
        costs = self._compute_costs(self.theta, self.W)

        # 3. Finite-difference gradient on z
        grad = np.zeros_like(self.z)
        eps = 0.01
        U0 = costs["U_total"]
        for k in range(min(len(self.z), 30)):  # Limit to 30 dims for speed
            z_plus = self.z.copy()
            z_plus[k] += eps
            W_plus = self._decode(z_plus)
            theta_tmp = self.theta.copy()
            for _ in range(3):  # Quick micro eval
                theta_tmp = self._micro_step(theta_tmp, W_plus)
            costs_plus = self._compute_costs(theta_tmp, W_plus)
            grad[k] = (costs_plus["U_total"] - U0) / eps

        # Add field pressure (bias toward connectivity)
        grad -= c.field_pressure * np.sign(self.z)

        # Update z
        self.z -= c.lr_z * grad

        # Re-decode W
        self.W = self._decode(self.z)
        eigvals, fiedler, spectral_gap = self._spectral(self.W)
        self.eigvals = eigvals

        self._outer_step += 1

        return SSGFState(
            z=self.z.copy(),
            W=self.W.copy(),
            theta=self.theta.copy(),
            eigvals=eigvals,
            R_global=costs["R_global"],
            fiedler=fiedler,
            spectral_gap=spectral_gap,
            C_micro=costs["C_micro"],
            U_total=costs["U_total"],
            outer_step=self._outer_step,
        )

    def get_audio_mapping(self) -> Dict[str, float]:
        """
        Map current SSGF state to audio parameters.

        Returns dict with CCW-compatible audio control values.
        """
        R = self._compute_R(self.theta)
        _, fiedler, spectral_gap = self._spectral(self.W)

        # Layer 2 phase velocity → binaural beat Hz
        if len(self.theta) > 1:
            dtheta_2 = float(abs(self.theta[1] - self.theta[0]))
            binaural_hz = 0.5 + (dtheta_2 / np.pi) * 39.5  # Map to 0.5-40 Hz
        else:
            binaural_hz = 10.0

        # Layer 4 coherence → isochronic pulse rate
        if len(self.theta) > 3:
            plv_34 = float(np.abs(np.exp(1j * (self.theta[2] - self.theta[3]))))
            pulse_rate = 1.0 + plv_34 * 15.0  # 1-16 Hz
        else:
            pulse_rate = 4.0

        # Layer 7 phase → spatial audio rotation angle
        spatial_angle = float(self.theta[min(6, len(self.theta) - 1)]) if len(self.theta) > 0 else 0.0

        theurgic_mode = R > 0.95

        return {
            "intensity": float(np.clip(R, 0, 1)),
            "binaural_hz": round(float(np.clip(binaural_hz, 0.5, 40.0)), 2),
            "pulse_rate": round(float(np.clip(pulse_rate, 1.0, 16.0)), 2),
            "spatial_angle": round(float(spatial_angle), 3),
            "fiedler": round(fiedler, 4),
            "spectral_gap": round(spectral_gap, 4),
            "theurgic_mode": bool(theurgic_mode),
            "R_global": round(float(R), 4),
        }

    def update_config(self, **kwargs):
        """Update config parameters (for adaptive feedback)."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def reset(self, seed: Optional[int] = None):
        """Reset engine state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.z = self.rng.normal(0, 0.1, self.config.latent_dim)
        self.theta = self.rng.uniform(0, 2 * np.pi, self.config.N)
        self.W = self._decode(self.z)
        self._outer_step = 0
