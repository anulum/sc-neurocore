# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — 1D heat equation via Feynman-Kac (Brownian motion expectation)

"""Feynman-Kac stochastic solver for the 1D heat equation.

Implementation reference:
  Karatzas, I. & Shreve, S. (1991). *Brownian Motion and
  Stochastic Calculus*, 2nd ed., Springer. §4.4 (Feynman-Kac
  formula for parabolic PDEs).
  Kloeden, P.E. & Platen, E. (1992). *Numerical Solution of
  Stochastic Differential Equations*. Springer. §10.5 (random
  walk approximations of diffusions).

The Feynman-Kac representation of the 1D heat equation
``∂u/∂t = α · ∂²u/∂x²`` with initial condition ``u(x, 0) = f(x)``
on the interval ``[0, L]`` with reflective boundaries is

    u(x, T) = E[ f(X_T) | X_0 = x ]

where ``X_t`` is a Brownian motion with variance ``2α·t``,
reflected at the boundaries ``0`` and ``L``.

This is **not** the same as a discrete random walk on an integer
lattice — the diffusion coefficient α MUST drive the per-step
variance, otherwise the discretisation simulates a different PDE.
The Antigravity placeholder previously here used integer steps
with `p=[0.25, 0.5, 0.25]` and never read `self.alpha` — that was
not Feynman-Kac and was replaced 2026-04-17 per
`feedback_sophisticated_from_start.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class FeynmanKacHeatSolver:
    """Solve the 1D heat equation via Feynman-Kac path-integral expectation.

    PDE: ``∂u/∂t = α · ∂²u/∂x²`` on ``x ∈ [0, L]`` with
         reflective BC ``u'(0,t) = u'(L,t) = 0``
         and initial condition ``u(x, 0) = f(x)``.

    Solution (Feynman-Kac):
        ``u(x_eval, T) = E[ f(X_T) | X_0 = x_eval ]``
    where ``X_t`` is reflected Brownian motion on ``[0, L]`` with
    variance ``2α t``.

    Parameters
    ----------
    length : float
        Domain extent L (same units as the diffusion coefficient
        squared per unit time). Defaults to a 1.0-unit interval.
    diffusivity : float
        α — the heat-equation diffusion coefficient (m²/s in SI;
        units are caller's responsibility).
    num_walkers : int
        Number of Monte Carlo walkers; estimator variance scales
        as 1/sqrt(N).
    dt : float
        Time step for the Euler-Maruyama integration of the
        Brownian motion. Must be small enough that
        ``sqrt(2 α dt) << L`` so reflection is well-resolved.
    seed : int
        RNG seed for reproducibility.
    """

    length: float = 1.0
    diffusivity: float = 1.0
    num_walkers: int = 10_000
    dt: float = 1e-3
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # Caller calls `set_initial_distribution` to seed the
        # walker positions; we leave them empty until then.
        self.walkers: np.ndarray = np.empty(0, dtype=np.float64)
        self._t: float = 0.0

    # ─────────────────────── walker initialisation ───────────────────────

    def set_initial_distribution(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        n_grid: int = 1024,
    ) -> None:
        """Sample walker positions from the initial distribution f(x).

        ``f`` must be a non-negative integrable function on ``[0, L]``.
        We discretise on a grid of ``n_grid`` cells (left edges at
        ``[0, L/n, 2L/n, ..., (n-1)L/n]``), normalise to a PMF, and
        inverse-transform sample. Sub-cell jitter is uniform across
        each cell's width, so a uniform PDF in maps to a uniform
        sample out (no boundary bias).
        """
        cell_w = self.length / n_grid
        # Use cell CENTRES for the f(x) evaluation so the Riemann sum
        # is the midpoint rule (second-order accurate).
        x_centres = (np.arange(n_grid) + 0.5) * cell_w
        pmf = np.maximum(f(x_centres), 0.0)
        total = pmf.sum()
        if total <= 0.0:
            raise ValueError("initial distribution f(x) must integrate to > 0")
        pmf = pmf / total
        cdf = np.cumsum(pmf)
        u = self._rng.random(self.num_walkers)
        idx = np.searchsorted(cdf, u, side="right")
        idx = np.clip(idx, 0, n_grid - 1)
        # x_grid here is the LEFT edge of each cell; jitter then
        # places walkers uniformly within the cell width.
        x_left = idx * cell_w
        jitter = self._rng.random(self.num_walkers) * cell_w
        self.walkers = np.clip(x_left + jitter, 0.0, self.length)
        self._t = 0.0

    def set_initial_delta(self, x_0: float) -> None:
        """All walkers start at position x_0 (delta-function initial condition)."""
        if not 0.0 <= x_0 <= self.length:
            raise ValueError(f"x_0={x_0} outside [0, {self.length}]")
        self.walkers = np.full(self.num_walkers, x_0, dtype=np.float64)
        self._t = 0.0

    # ─────────────────────── Brownian-motion stepping ────────────────────

    def step(self, n_substeps: int = 1) -> None:
        """Advance walkers by `n_substeps` Brownian steps.

        Each step:
            X_{t+dt} = X_t + sqrt(2α dt) · N(0, 1)
        followed by reflection at the boundaries 0 and L.
        """
        if self.walkers.size == 0:
            raise RuntimeError(
                "walkers not initialised; call "
                "set_initial_distribution() or set_initial_delta() first"
            )
        sigma = np.sqrt(2.0 * self.diffusivity * self.dt)
        for _ in range(n_substeps):
            increments = self._rng.standard_normal(self.walkers.size) * sigma
            self.walkers += increments
            # Reflect at boundaries (mirror image)
            # Repeat until all walkers are in [0, L] — usually one
            # iteration suffices for σ << L.
            for _safety in range(64):
                under = self.walkers < 0.0
                over = self.walkers > self.length
                if not (under.any() or over.any()):
                    break
                self.walkers[under] = -self.walkers[under]
                self.walkers[over] = 2.0 * self.length - self.walkers[over]
        self._t += n_substeps * self.dt

    def evolve_to(self, T: float) -> None:
        """Step walkers from current time to ``T`` (T > current t)."""
        if T < self._t:
            raise ValueError(f"T={T} < current t={self._t}; cannot run backwards")
        n = int(round((T - self._t) / self.dt))
        if n > 0:
            self.step(n)

    # ─────────────────────── solution sampling ───────────────────────────

    def get_density(self, n_bins: int = 64) -> np.ndarray[Any, Any]:
        """Return the Monte Carlo histogram density over [0, L].

        Each bin reports the count of walkers in that sub-interval
        normalised so that the integral over [0, L] equals 1
        (i.e. it is a probability density).
        """
        if self.walkers.size == 0:
            raise RuntimeError("walkers not initialised")
        counts, _ = np.histogram(
            self.walkers, bins=n_bins, range=(0.0, self.length)
        )
        bin_width = self.length / n_bins
        return counts.astype(np.float64) / (self.walkers.size * bin_width)

    def expectation(
        self,
        observable: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Compute the Feynman-Kac expectation `E[observable(X_T)]`.

        For an initial-value problem with `f` as the initial
        condition and walkers placed via `set_initial_delta(x_0)`,
        the expectation `E[f(X_T)]` equals `u(x_0, T)`.
        """
        if self.walkers.size == 0:
            raise RuntimeError("walkers not initialised")
        return float(np.mean(observable(self.walkers)))

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._t


# Backwards-compatibility alias — earlier code used the old name.
# The old class did not actually implement Feynman-Kac, so the
# alias is a name-only redirect; behaviour now matches the proper
# stochastic solver.
StochasticHeatSolver = FeynmanKacHeatSolver
