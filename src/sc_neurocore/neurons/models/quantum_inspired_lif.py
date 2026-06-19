# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-Inspired LIF Neuron

"""Quantum-inspired LIF neuron with non-classical probability logic.

Extends standard LIF by maintaining a complex-valued amplitude z = a + bi
whose squared modulus |z|**2 determines the firing probability. Interference
between excitatory and inhibitory inputs can produce non-classical
suppression patterns (destructive interference).

Equations:

    dz/dt = (-z + I_complex) / tau
    P(spike) = |z|**2 / theta**2

Stochastic spike: a uniform random draw < P(spike) triggers a spike
with reset z -> v_reset. Uses xorshift64 PRNG for reproducibility.

Reference: Quantum-neural hybrid models, IBM Heron r2 noise models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class QuantumInspiredLIFNeuron:
    """Quantum-inspired LIF with complex amplitude and stochastic firing.

    Parameters
    ----------
    tau : float
        Membrane time constant (ms). Default: 20.0.
    theta : float
        Firing threshold for |z|. Default: 1.0.
    dt : float
        Integration timestep (ms). Default: 0.1.
    v_reset : float
        Reset value for z_re and z_im after spike. Default: 0.0.
    seed : int
        Initial RNG state for xorshift64. Default: 12345.
    """

    tau: float = 20.0
    theta: float = 1.0
    dt: float = 0.1
    v_reset: float = 0.0
    seed: int = 12345

    z_re: float = field(default=0.0, repr=False)
    z_im: float = field(default=0.0, repr=False)
    _rng_state: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        for name in ("tau", "theta", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        for name in ("v_reset", "z_re", "z_im"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if type(self.seed) is not int or not (0 < self.seed < 2**64):
            raise ValueError("seed must be an integer in [1, 2**64)")

        self._rng_state = self.seed

    def _xorshift64(self) -> float:
        """Xorshift64 PRNG returning uniform in [0, 1)."""
        x = self._rng_state & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self._rng_state = x
        return (x & 0xFFFFFFFF) / 4294967296.0

    def step_complex(self, i_re: float, i_im: float) -> int:
        """Step with real and imaginary current components.

        Returns 1 if stochastic spike, 0 otherwise.
        """
        if not math.isfinite(i_re) or not math.isfinite(i_im):
            raise ValueError("current components must be finite")

        dz_re = (-self.z_re + i_re) / self.tau
        dz_im = (-self.z_im + i_im) / self.tau
        self.z_re += dz_re * self.dt
        self.z_im += dz_im * self.dt

        prob = (self.z_re**2 + self.z_im**2) / (self.theta**2)
        uniform = self._xorshift64()

        if uniform < min(prob, 1.0):
            self.z_re = self.v_reset
            self.z_im = self.v_reset
            return 1
        return 0

    def step(self, current: float) -> int:
        """Step with real-only current (imaginary = 0)."""
        return self.step_complex(current, 0.0)

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.z_re = 0.0
        self.z_im = 0.0
        self._rng_state = self.seed
