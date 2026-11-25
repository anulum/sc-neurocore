# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel & Hakim 1999 — Ornstein-Uhlenbeck driven IF

"""Integrate-and-fire neuron driven by Gaussian white noise, from its own seeded generator or NumPy's global one."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StochasticIFNeuron:
    """Brunel & Hakim 1999 — Ornstein-Uhlenbeck driven IF.

    Reference: Tuckwell, H.C. (1988). Introduction to Theoretical Neurobiology, Vol. 2. Cambridge Univ. Press.

    With ``seed`` set, the noise comes from the neuron's own ``numpy.random.Generator``
    seeded with it, so two neurons with the same seed give the same spike train whatever
    else draws random numbers, and ``reset`` restarts the stream. Without a seed the noise
    comes from NumPy's process-wide generator (``np.random.randn``), as before, so
    ``np.random.seed`` still governs it.
    """

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    mu: float = 0.0
    sigma: float = 3.0
    dt: float = 1.0
    seed: int | None = None
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse non-finite voltages and drift, non-positive time constants, negative sigma and a bad seed."""
        for name in ("v", "v_rest", "v_reset", "v_threshold", "mu"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("tau_m", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError("sigma must be non-negative and finite")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an int or None")
            if self.seed < 0:
                raise ValueError("seed must be non-negative")
        self._rng = None if self.seed is None else np.random.default_rng(self.seed)

    def step(self, current: float) -> int:
        """Advance one Euler-Maruyama step with one standard-normal draw; return the spike.

        The draw comes from the neuron's own generator when ``seed`` is set, otherwise from
        ``np.random.randn()``.

        Parameters
        ----------
        current : float
            Finite input current of this step.

        Returns
        -------
        int
            1 when the voltage reached ``v_threshold`` (and was reset), else 0.
        """
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        draw = np.random.randn() if self._rng is None else self._rng.standard_normal()
        noise = self.sigma * np.sqrt(self.dt / self.tau_m) * draw
        self.v += (-(self.v - self.v_rest) + self.mu + current) / self.tau_m * self.dt + noise
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        """Return the membrane voltage to ``v_rest``; a seeded neuron also restarts its noise stream."""
        self.v = self.v_rest
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)
