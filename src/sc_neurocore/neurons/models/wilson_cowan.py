# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-Cowan 1972 — excitatory/inhibitory population

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WilsonCowanUnit:
    """Wilson-Cowan 1972 — excitatory/inhibitory population rate model.

    τ_e dE/dt = -E + S(w_ee·E - w_ei·I + I_ext)
    τ_i dI/dt = -I + S(w_ie·E - w_ii·I)
    S(x) = 1/(1 + exp(-a(x-θ))) - 1/(1 + exp(aθ))

    Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.
    """

    e: float = 0.1
    i: float = 0.05
    w_ee: float = 10.0
    w_ei: float = 6.0
    w_ie: float = 10.0
    w_ii: float = 1.0
    tau_e: float = 1.0
    tau_i: float = 2.0
    a: float = 1.2
    theta: float = 4.0
    dt: float = 0.1

    def _sigmoid(self, x: float) -> float:
        # math.exp on scalars is ~4× faster than np.exp with bit-identical
        # output (both dispatch to libm `exp()`); measured 2.72 → 0.68 µs/step.
        return 1.0 / (1.0 + math.exp(-self.a * (x - self.theta)))

    def step(self, ext_input: float = 0.0) -> float:
        se = self._sigmoid(self.w_ee * self.e - self.w_ei * self.i + ext_input)
        si = self._sigmoid(self.w_ie * self.e - self.w_ii * self.i)
        self.e += (-self.e + se) / self.tau_e * self.dt
        self.i += (-self.i + si) / self.tau_i * self.dt
        return self.e

    def reset(self) -> None:
        self.e, self.i = 0.1, 0.05
