# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Amari 1977 — continuous neural field, discretized on N nodes

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class AmariNeuralField:
    """Amari 1977 — continuous neural field, discretized on N nodes.

    tau du_i/dt = -u_i + sum_j w(|i-j|) f(u_j) * dx + I_i
    w(x) = A * exp(-a*|x|) - B * exp(-b*|x|)    (Mexican hat)
    f(u) = max(0, u)                              (Heaviside-linear)
    """

    n: int = 64
    tau: float = 10.0
    a_exc: float = 1.5
    a_width: float = 1.0
    b_inh: float = 0.75
    b_width: float = 2.0
    dx: float = 0.5
    dt: float = 0.5
    u: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    _w: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.u is None:
            self.u = np.zeros(self.n)
        self._build_kernel()

    def _build_kernel(self) -> None:
        x = np.abs(np.arange(self.n) - self.n // 2) * self.dx
        k = self.a_exc * np.exp(-self.a_width * x) - self.b_inh * np.exp(-self.b_width * x)
        self._w = np.roll(k, -self.n // 2)

    def step(self, current: NDArray[Any]) -> float:
        """Advance one timestep. Returns mean activation."""
        f_u = np.maximum(self.u, 0.0)
        conv = np.real(np.fft.ifft(np.fft.fft(self._w) * np.fft.fft(f_u))) * self.dx
        self.u += (-self.u + conv + current) / self.tau * self.dt
        return float(np.mean(np.maximum(self.u, 0.0)))

    def reset(self) -> None:
        self.u = np.zeros(self.n)
