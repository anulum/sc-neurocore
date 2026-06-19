# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — L16 Director: self-monitoring controller for identity substrate

"""SCPN Layer 16: cybernetic closure for the identity substrate.

Monitors network health, applies corrective actions when dynamics
drift from healthy bounds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.analysis import (
    cv_isi,
    fano_factor,
    firing_rate,
    permutation_entropy,
)

from .substrate import IdentitySubstrate

RATE_BOUNDS = (5.0, 20.0)  # Hz, cortical physiological range
CV_BOUNDS = (0.5, 1.5)  # CV ~ 1.0 for Poisson-like firing
FANO_BOUNDS = (0.5, 2.0)  # near-Poisson variability
PRUNE_THRESHOLD = 0.01  # weights below this get zeroed
GROW_FRACTION = 0.02  # fraction of zero weights to reinitialize


class DirectorController:
    """L16 self-monitoring and self-regulation for the identity substrate."""

    def __init__(self, substrate: IdentitySubstrate):
        self.substrate = substrate
        self.target_rate = RATE_BOUNDS
        self.target_cv = CV_BOUNDS
        self.target_fano = FANO_BOUNDS
        self._corrections_applied = 0

    def monitor(self) -> dict[str, Any]:
        """Measure current dynamics from recent spike history."""
        history = self.substrate.spike_history
        if len(history) < 50:
            return {
                "mean_rate": 0.0,
                "cv": float("nan"),
                "fano": float("nan"),
                "perm_entropy": float("nan"),
                "n_steps": len(history),
            }

        recent = np.array(history[-500:], dtype=np.int8)
        pop_binary = (recent.sum(axis=1) > 0).astype(np.int8)

        return {
            "mean_rate": firing_rate(pop_binary),
            "cv": cv_isi(pop_binary),
            "fano": fano_factor(pop_binary, window_ms=50.0),
            "perm_entropy": permutation_entropy(pop_binary),
            "n_steps": len(history),
        }

    def diagnose(self) -> list[str]:
        """Identify problems in network dynamics."""
        metrics = self.monitor()
        problems = []

        rate = metrics["mean_rate"]
        if rate > self.target_rate[1]:
            problems.append("rate_too_high")
        elif rate < self.target_rate[0] and rate > 0:
            problems.append("rate_too_low")
        elif rate == 0 and metrics["n_steps"] > 100:
            problems.append("silent")

        cv = metrics["cv"]
        if not np.isnan(cv):
            if cv < self.target_cv[0]:
                problems.append("too_regular")
            elif cv > self.target_cv[1]:
                problems.append("too_chaotic")

        fano = metrics["fano"]
        if not np.isnan(fano):
            if fano > self.target_fano[1]:
                problems.append("bursty")

        ee_weights = self.substrate.proj_ee.data
        density = np.count_nonzero(ee_weights) / max(ee_weights.size, 1)
        if density > 0.95:
            problems.append("connectivity_too_dense")
        elif density < 0.05 and ee_weights.size > 0:
            problems.append("connectivity_too_sparse")

        return problems

    def correct(self) -> None:
        """Apply corrective actions based on diagnosis."""
        problems = self.diagnose()
        if not problems:
            return

        for problem in problems:
            if problem == "rate_too_high":
                self.substrate.proj_ie.data *= 1.1
            elif problem in ("rate_too_low", "silent"):
                self.substrate.proj_ie.data *= 0.9
            elif problem == "too_regular":
                _add_weight_noise(self.substrate.proj_ee.data, scale=0.05)
            elif problem == "too_chaotic":
                _homeostatic_scale(self.substrate.proj_ee.data, factor=0.95)
            elif problem == "bursty":
                self.substrate.proj_ie.data *= 1.05
                self.substrate.proj_ii.data *= 1.05
            elif problem == "connectivity_too_dense":
                _prune_weak(self.substrate.proj_ee.data, PRUNE_THRESHOLD)
            elif problem == "connectivity_too_sparse":
                _grow_synapses(self.substrate.proj_ee.data, GROW_FRACTION, self.substrate.seed)

        self._corrections_applied += 1

    def report(self) -> str:
        """Generate human-readable health report."""
        metrics = self.monitor()
        problems = self.diagnose()

        lines = [
            f"Rate: {metrics['mean_rate']:.1f} Hz (target: {self.target_rate[0]}-{self.target_rate[1]})",
            f"CV: {metrics['cv']:.2f} (target: {self.target_cv[0]}-{self.target_cv[1]})",
            f"Fano: {metrics['fano']:.2f} (target: {self.target_fano[0]}-{self.target_fano[1]})",
            f"Permutation entropy: {metrics['perm_entropy']:.3f}",
            f"Corrections applied: {self._corrections_applied}",
        ]

        if problems:
            lines.append(f"Diagnosis: {', '.join(problems)}")
        else:
            lines.append("Diagnosis: healthy")

        return "\n".join(lines)


def _add_weight_noise(data: np.ndarray[Any, Any], scale: float) -> None:
    """Add Gaussian noise to nonzero weights, clip to non-negative."""
    mask = data > 0
    noise = np.random.default_rng().normal(0, scale, size=data.shape)
    data[mask] += noise[mask]
    np.clip(data, 0, None, out=data)


def _homeostatic_scale(data: np.ndarray[Any, Any], factor: float) -> None:
    """Scale all weights toward the mean. Turrigiano 2008."""
    mean_w = data[data > 0].mean() if np.any(data > 0) else 0.0
    if mean_w > 0:
        data[:] = mean_w + factor * (data - mean_w)
        np.clip(data, 0, None, out=data)


def _prune_weak(data: np.ndarray[Any, Any], threshold: float) -> None:
    """Zero out weights below threshold."""
    data[data < threshold] = 0.0


def _grow_synapses(data: np.ndarray[Any, Any], fraction: float, seed: int) -> None:
    """Reinitialize a fraction of zero weights to small positive values."""
    rng = np.random.default_rng(seed)
    zero_mask = data == 0.0
    n_zeros = zero_mask.sum()
    n_grow = max(1, int(n_zeros * fraction))
    if n_zeros == 0:
        return
    indices = np.where(zero_mask)[0]
    chosen = rng.choice(indices, size=min(n_grow, len(indices)), replace=False)
    data[chosen] = rng.uniform(0.01, 0.1, size=chosen.shape)
