# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chaotic RNG for Stochastic Computing Bitstreams

"""Chaotic random number generators for stochastic computing bitstreams.

Provides deterministic-chaos alternatives to linear PRNGs (LFSR, MT19937)
for SC bitstream encoding. Chaotic maps produce sequences with desirable
statistical properties for stochastic arithmetic: uniform marginal
distribution (at r=4), low short-range autocorrelation, and tunable
correlation structure.

References:
    Phatak & Rao, "Logistic map as a random number generator",
    Physical Review E 51(4), 1995.

    Li et al., "Chaotic random bit generator realized with a
    microcontroller", J. Micromech. Microeng., 2019.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ChaoticRNG:
    """Logistic-map chaotic RNG for SC bitstream generation.

    x_{n+1} = r * x_n * (1 - x_n)

    At r=4.0 the logistic map is fully chaotic with Lyapunov exponent
    ln(2) ~ 0.693 and an invariant density Beta(0.5, 0.5) on (0, 1).
    The 100-step burn-in discards transients from the initial condition.

    Parameters
    ----------
    r : float
        Bifurcation parameter. Must be in (3.57, 4.0] for chaos.
        Default 4.0 gives maximal chaos.
    x : float
        Initial condition in (0, 1). Avoid 0.0, 0.5, 1.0 exactly
        (these are fixed/periodic points at r=4).
    burn_in : int
        Steps to discard before first output.

    Example
    -------
    >>> rng = ChaoticRNG(r=4.0, x=0.37)
    >>> bits = rng.generate_bitstream(p=0.5, length=1000)
    >>> 0.4 < bits.mean() < 0.6
    True
    """

    r: float = 4.0
    x: float = 0.37
    burn_in: int = 100
    _state: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.x < 1.0:
            raise ValueError(f"x must be in (0, 1), got {self.x}")
        if not 3.0 < self.r <= 4.0:
            raise ValueError(f"r must be in (3, 4], got {self.r}")
        self._state = self.x
        for _ in range(self.burn_in):
            self._state = self.r * self._state * (1.0 - self._state)

    def random(self, size: int) -> np.ndarray[Any, Any]:
        """Generate *size* chaotic floats in (0, 1).

        Uses scalar iteration (logistic map is inherently sequential).
        For bulk generation, see :meth:`random_vectorized`.
        """
        out = np.empty(size, dtype=np.float64)
        s = self._state
        r = self.r
        for i in range(size):
            s = r * s * (1.0 - s)
            out[i] = s
        self._state = s
        return out

    def random_vectorized(self, size: int, n_maps: int = 8) -> np.ndarray[Any, Any]:
        """Generate samples from *n_maps* independent logistic maps in parallel.

        Seeds each map with a different initial condition derived from the
        primary state. Returns *size* samples by round-robin interleaving.
        Faster than scalar iteration for large *size* at the cost of
        slightly different correlation structure.
        """
        states = np.empty(n_maps, dtype=np.float64)
        s = self._state
        for j in range(n_maps):
            s = self.r * s * (1.0 - s)
            states[j] = s
        self._state = s

        steps_per_map = (size + n_maps - 1) // n_maps
        buf = np.empty((n_maps, steps_per_map), dtype=np.float64)
        for t in range(steps_per_map):
            states = self.r * states * (1.0 - states)  # type: ignore[assignment]
            buf[:, t] = states
        self._state = float(states[0])
        return buf.ravel(order="F")[:size]

    def generate_bitstream(self, p: float, length: int) -> np.ndarray[Any, Any]:
        """Generate SC bitstream where P(bit=1) ~ *p*.

        Applies the inverse CDF of Beta(0.5, 0.5) to map chaotic samples to
        uniform, then thresholds at *p*. This corrects for the non-uniform
        invariant density of the logistic map at r=4.

        Parameters
        ----------
        p : float
            Target probability in [0, 1].
        length : int
            Number of bits.

        Returns
        -------
        np.ndarray[Any, Any]
            Binary array of shape (length,), dtype uint8.
        """
        vals = self.random(length)
        # CDF of Beta(0.5,0.5) is (2/pi)*arcsin(sqrt(x)) — apply to uniformize
        uniform = np.arcsin(np.sqrt(np.clip(vals, 1e-15, 1.0 - 1e-15))) * (2.0 / np.pi)
        bits: np.ndarray[Any, Any] = (uniform < p).astype(np.uint8)
        return bits

    def lyapunov_exponent(self, n_steps: int = 10_000) -> float:
        """Estimate the maximal Lyapunov exponent via derivative averaging.

        For the logistic map: lambda = (1/N) * sum(ln|r*(1 - 2*x_n)|).
        At r=4.0, the theoretical value is ln(2) ~ 0.6931.
        """
        s = self._state
        r = self.r
        total = 0.0
        for _ in range(n_steps):
            deriv = abs(r * (1.0 - 2.0 * s))
            if deriv > 0:
                total += np.log(deriv)
            s = r * s * (1.0 - s)
        self._state = s
        return total / n_steps

    def shannon_entropy(self, n_samples: int = 10_000, n_bins: int = 100) -> float:
        """Estimate Shannon entropy of the chaotic sequence in bits.

        Uniform distribution on (0, 1) has entropy log2(n_bins).
        At r=4.0, values follow Beta(0.5, 0.5) with theoretical
        entropy ~ log2(n_bins) - 0.27 bits below uniform.
        """
        samples = self.random(n_samples)
        counts, _ = np.histogram(samples, bins=n_bins, range=(0.0, 1.0))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def autocorrelation(self, n_samples: int = 10_000, max_lag: int = 50) -> np.ndarray[Any, Any]:
        """Compute autocorrelation of the chaotic sequence up to *max_lag*.

        A good chaotic RNG should show near-zero autocorrelation for all
        lags > 0. Returns array of shape (max_lag + 1,) with R[0] = 1.0.
        """
        samples = self.random(n_samples)
        mean = samples.mean()
        var = samples.var()
        if var == 0:  # pragma: no cover
            return np.zeros(max_lag + 1)
        centered = samples - mean
        acf = np.empty(max_lag + 1, dtype=np.float64)
        acf[0] = 1.0
        for lag in range(1, max_lag + 1):
            acf[lag] = np.dot(centered[: n_samples - lag], centered[lag:]) / (
                (n_samples - lag) * var
            )
        return acf

    def reset(self, x: float | None = None) -> None:
        """Reset to initial condition (with fresh burn-in)."""
        self._state = x if x is not None else self.x
        for _ in range(self.burn_in):
            self._state = self.r * self._state * (1.0 - self._state)

    @property
    def state(self) -> float:
        """Current internal state."""
        return self._state


@dataclass
class TentMapRNG:
    """Tent map chaotic RNG — piecewise linear alternative to logistic map.

    x_{n+1} = mu * min(x_n, 1 - x_n)

    At mu=2.0 the tent map is topologically conjugate to the logistic map
    at r=4.0 but has uniform invariant density on (0, 1) — better for
    SC bitstream generation where uniform marginals are desired.

    Parameters
    ----------
    mu : float
        Slope parameter. Must be in (1, 2] for chaos. Default 2.0.
    x : float
        Initial condition in (0, 1).
    """

    # mu=2.0 exactly is degenerate in float64 (hits rational periodic orbits).
    # mu=1.9999 retains full chaos with uniform-like density.
    mu: float = 1.9999
    x: float = 0.37
    burn_in: int = 100
    _state: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.x < 1.0:
            raise ValueError(f"x must be in (0, 1), got {self.x}")
        if not 1.0 < self.mu <= 2.0:
            raise ValueError(f"mu must be in (1, 2], got {self.mu}")
        self._state = self.x
        for _ in range(self.burn_in):
            self._state = self._step(self._state)

    def _step(self, s: float) -> float:
        s = self.mu * min(s, 1.0 - s)
        # Guard against collapse to 0 (fixed point at mu=2.0)
        if s < 1e-15:
            s = 1e-10
        return s

    def random(self, size: int) -> np.ndarray[Any, Any]:
        """Generate *size* chaotic floats in (0, 1)."""
        out = np.empty(size, dtype=np.float64)
        s = self._state
        for i in range(size):
            s = self._step(s)
            out[i] = s
        self._state = s
        return out

    def generate_bitstream(self, p: float, length: int) -> np.ndarray[Any, Any]:
        """Generate SC bitstream where P(bit=1) ~ *p*."""
        vals = self.random(length)
        return (vals < p).astype(np.uint8)

    def reset(self, x: float | None = None) -> None:
        """Reset to initial condition (with fresh burn-in)."""
        self._state = x if x is not None else self.x
        for _ in range(self.burn_in):
            self._state = self._step(self._state)

    @property
    def state(self) -> float:
        return self._state
