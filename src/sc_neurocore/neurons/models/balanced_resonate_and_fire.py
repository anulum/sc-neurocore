# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Balanced Resonate-and-Fire Neuron

"""Balanced Resonate-and-Fire neuron from Higuchi et al. (ICML 2024).

The implementation follows Algorithm 1 in "Balanced Resonate-and-Fire
Neurons": refractory threshold, smooth reset through the damping term, and the
discrete-time divergence boundary ``p(omega)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def sustain_oscillation_boundary(omega: float, dt: float) -> float:
    """Return the BRF divergence boundary ``p(omega)``.

    ``p(omega) = (-1 + sqrt(1 - (dt * omega)^2)) / dt``.
    The value is real only when ``0 < dt * omega <= 1``.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if omega <= 0.0:
        raise ValueError("omega must be positive.")
    scaled = dt * omega
    if scaled > 1.0:
        raise ValueError(
            "BRF divergence boundary is undefined for dt * omega > 1; "
            f"got dt={dt!r}, omega={omega!r}."
        )
    return (-1.0 + math.sqrt(max(0.0, 1.0 - scaled * scaled))) / dt


@dataclass
class BalancedResonateAndFireNeuron:
    """Balanced RF neuron with refractory threshold and smooth reset.

    State variables follow the paper notation ``u = x + i y`` and refractory
    state ``q``. One scalar update computes:

    ``b_t = p(omega) - b_offset - q_{t-1}``

    ``u_t = u_{t-1} + dt * ((b_t + i * omega) * u_{t-1} + current)``

    ``theta_t = theta_c + q_{t-1}``

    ``z_t = Heaviside(Re(u_t) - theta_t)``

    ``q_t = gamma * q_{t-1} + z_t``

    Reference: Higuchi, Kairat, Bohte, and Otte (2024), "Balanced
    Resonate-and-Fire Neurons", Proceedings of ICML 2024, Algorithm 1.
    """

    x: float = 0.0
    y: float = 0.0
    q: float = 0.0
    omega: float = 10.0
    b_offset: float = 1.0
    threshold: float = 1.0
    gamma: float = 0.9
    dt: float = 0.01

    def __post_init__(self) -> None:
        self._validate_parameters()

    @property
    def p_omega(self) -> float:
        """Current divergence boundary for ``omega`` and ``dt``."""
        return sustain_oscillation_boundary(self.omega, self.dt)

    @property
    def damping(self) -> float:
        """Current smooth-reset damping ``b_t`` before the next step."""
        return self.p_omega - self.b_offset - self.q

    @property
    def dynamic_threshold(self) -> float:
        """Current refractory threshold ``theta_c + q``."""
        return self.threshold + self.q

    def step(self, current: float) -> int:
        """Advance one BRF timestep and return the binary spike ``z_t``."""
        self._validate_parameters()
        b_t = self.damping
        theta_t = self.dynamic_threshold

        x_prev = self.x
        y_prev = self.y
        self.x = x_prev + self.dt * (b_t * x_prev - self.omega * y_prev + current)
        self.y = y_prev + self.dt * (self.omega * x_prev + b_t * y_prev)

        spike = int(self.x >= theta_t)
        self.q = self.gamma * self.q + float(spike)
        return spike

    def reset(self) -> None:
        """Reset membrane and refractory state to rest."""
        self.x = 0.0
        self.y = 0.0
        self.q = 0.0

    def state(self) -> dict[str, float]:
        """Return a compact state snapshot for reproducibility tests."""
        return {
            "x": self.x,
            "y": self.y,
            "q": self.q,
            "omega": self.omega,
            "b_offset": self.b_offset,
            "threshold": self.threshold,
            "gamma": self.gamma,
            "dt": self.dt,
            "p_omega": self.p_omega,
            "damping": self.damping,
            "dynamic_threshold": self.dynamic_threshold,
        }

    def _validate_parameters(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.omega <= 0.0:
            raise ValueError("omega must be positive.")
        if self.dt * self.omega > 1.0:
            raise ValueError(
                "BRF requires dt * omega <= 1 so p(omega) remains real; "
                f"got dt={self.dt!r}, omega={self.omega!r}."
            )
        if self.b_offset <= 0.0:
            raise ValueError("b_offset must be positive.")
        if self.threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1.")
        for name, value in {
            "x": self.x,
            "y": self.y,
            "q": self.q,
            "omega": self.omega,
            "b_offset": self.b_offset,
            "threshold": self.threshold,
            "gamma": self.gamma,
            "dt": self.dt,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
