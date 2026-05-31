# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson 1999 — polynomial cortical model

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class WilsonHRNeuron:
    """Wilson 1999 — polynomial cortical model.

    dV/dt = -(17.81 + 47.71*V + 32.63*V^2)*(V - 0.55) - 26*R*(V + 0.92) + I
    dR/dt = (-R + 1.35*V + 1.03) / tau_R

    V in dimensionless units, spike at V > V_peak.

    Reference: Wilson, H.R. (1999). Spikes, Decisions, and Actions. Oxford Univ. Press.
    """

    v: float = -0.7
    r: float = 0.1
    tau_r: float = 1.9
    v_peak: float = 0.4
    dt: float = 0.05

    def __post_init__(self) -> None:
        for name in ("v", "r", "tau_r", "v_peak", "dt"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.tau_r <= 0.0:
            raise ValueError("tau_r must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    @staticmethod
    def _validate_state(v: float, r: float) -> tuple[float, float]:
        voltage = float(v)
        recovery = float(r)
        if not math.isfinite(voltage) or not math.isfinite(recovery):
            raise FloatingPointError("Wilson-HR runtime state must be finite")
        return voltage, recovery

    def step(self, current: float) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")

        v, r = self._validate_state(self.v, self.r)
        try:
            poly = -(17.81 + 47.71 * v + 32.63 * v**2) * (v - 0.55)
        except OverflowError as exc:
            raise FloatingPointError("Wilson-HR polynomial overflowed") from exc
        syn = -26.0 * r * (v + 0.92)
        dv = (poly + syn + drive) * self.dt
        dr = (-r + 1.35 * v + 1.03) / self.tau_r * self.dt
        next_v = v + dv
        next_r = r + dr
        if (
            not math.isfinite(poly)
            or not math.isfinite(syn)
            or not math.isfinite(dv)
            or not math.isfinite(dr)
            or not math.isfinite(next_v)
            or not math.isfinite(next_r)
        ):
            raise FloatingPointError("Wilson-HR candidate state became non-finite")

        self.v = next_v
        self.r = next_r
        if next_v >= self.v_peak:
            self.v = -0.7
            return 1
        return 0

    def reset(self) -> None:
        self.v = -0.7
        self.r = 0.1
