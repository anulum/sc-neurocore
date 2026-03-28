# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rall 1962 — N-compartment passive cable discretization

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RallCableNeuron:
    """Rall 1962 — N-compartment passive cable discretization.

    Each compartment: C dV_i/dt = -g_L(V_i - E_L) + g_a(V_{i-1} - 2V_i + V_{i+1})
    Soma is compartment 0; input injected at distal end (N-1).
    Spike detection at soma.
    """

    n_comp: int = 5
    tau_m: float = 20.0  # ms
    v_rest: float = -65.0  # mV
    g_ratio: float = 0.5  # g_axial / g_leak ratio (electrotonic coupling)
    v_threshold: float = -50.0
    v_reset: float = -65.0
    dt: float = 0.1
    v: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.v = np.full(self.n_comp, self.v_rest)

    def step(self, current: float) -> int:
        v_prev_soma = self.v[0]
        dv = np.zeros(self.n_comp)
        for i in range(self.n_comp):
            leak = -(self.v[i] - self.v_rest)
            left = self.v[i - 1] if i > 0 else self.v[i]
            right = self.v[i + 1] if i < self.n_comp - 1 else self.v[i]
            axial = self.g_ratio * (left - 2.0 * self.v[i] + right)
            inj = current if i == self.n_comp - 1 else 0.0
            dv[i] = (leak + axial + inj) / self.tau_m
        self.v += dv * self.dt
        if self.v[0] >= self.v_threshold and v_prev_soma < self.v_threshold:
            self.v[0] = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v[:] = self.v_rest
