# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SSGF Engine -- Lightweight Stochastic Synthesis of

"""
SSGF geometry-to-audio solver.

Pure-NumPy solver that couples Kuramoto phase oscillators with a
learned geometry matrix, producing real-time audio-mapping observables
for binaural frequency, pulse rate, spatial angle, intensity, and
spectral connectivity.

The implementation is self-contained and intentionally dependency-light:
it uses NumPy plus the SC-NeuroCore parameter table so the adaptive audio
controller can run without JAX, PyTorch, or topology packages.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from sc_neurocore.scpn.params import OMEGA_N, build_knm_matrix

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class SSGFConfig:
    """Configuration for the SSGF geometry-coupled oscillator engine.

    Attributes
    ----------
    N:
        Number of oscillators in the Kuramoto field.
    z_dim:
        Length of the latent geometry vector decoded into the symmetric
        coupling matrix.
    lr_z:
        Gradient-descent step size for the latent geometry vector.
    sigma_g:
        Scale applied to geometry-derived phase coupling.
    micro_steps:
        Number of Kuramoto integration steps per outer geometry update.
    dt:
        Integration timestep in seconds.
    noise:
        Standard deviation of phase noise injected during each micro-step.
    K_base:
        Baseline Kuramoto coupling retained for compatibility with profile
        tuning surfaces.
    K_alpha:
        Adaptive coupling multiplier retained for compatibility with profile
        tuning surfaces.
    field_pressure:
        Cosine field pressure applied as a global steering term.
    seed:
        Deterministic NumPy random seed for reproducible initial conditions.
    """

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

    def __init__(self, cfg: SSGFConfig | None = None) -> None:
        """Initialise the SSGF state from a deterministic configuration.

        Parameters
        ----------
        cfg:
            Optional engine configuration. When omitted, ``SSGFConfig`` supplies
            the default oscillator count, integration constants, and seed.
        """
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
        self.K = build_knm_matrix(c.N)

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

    def _decode(self, z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Decode a latent vector into a symmetric non-negative weight matrix.

        The decoder tiles or truncates ``z`` across the upper-triangular shell,
        mirrors the shell, applies a stable softplus transform, and forces a
        zero diagonal so the geometry matrix has no self-coupling.
        """
        N = self.N
        # Number of unique off-diagonal upper-triangle entries
        n_upper = N * (N - 1) // 2
        # Tile z to fill if z_dim < n_upper, or truncate
        flat = np.tile(z, (n_upper // len(z) + 1))[:n_upper]

        A = np.zeros((N, N))
        idx_upper = np.triu_indices(N, k=1)
        A[idx_upper] = flat
        A = A + A.T  # type: ignore[assignment]  # symmetric

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
        self._eigvals = eigvals  # type: ignore[assignment]
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
        return float(c_micro + c_reg)

    # ── Outer Cycle ──────────────────────────────────────────────────

    def outer_step(self) -> float:
        """Advance one SSGF outer cycle.

        The step runs the configured Kuramoto micro-cycle, refreshes the
        spectral bridge, estimates a finite-difference gradient for the latent
        geometry vector, and stores the cost history used by state snapshots.

        Returns
        -------
        float
            Composite cost before the latent-vector update.
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

    def get_audio_mapping(self) -> dict[str, float | bool]:
        """Derive CCW audio parameters from current SSGF state.

        Returns
        -------
        dict[str, float | bool]
            JSON-compatible audio mapping containing ``binaural_hz``,
            ``pulse_rate``, ``spatial_angle``, ``intensity``, ``fiedler``,
            ``spectral_gap``, and ``theurgic_mode``.
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

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot of the current SSGF state.

        Returns
        -------
        dict[str, Any]
            Snapshot containing step counters, coherence, phase vector,
            geometry statistics, spectral values, latest cost, and the current
            audio mapping.
        """
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
