# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cerebellar Lugaro Cell

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LugaroCell:
    """Cerebellar Lugaro cell — rare fusiform granular layer interneuron.

    LIF with adaptation, serotonin (5-HT) modulation, depolarised leak
    for spontaneous firing. Inhibits Golgi cells and molecular layer INs.

    Reference: Dieudonné & Bhatt (2003) J Physiol 548:97;
    Lainé & Bhatt (2007) Front Syst Neurosci 1:4.
    """

    v: float = -55.0
    adapt: float = 0.0
    v_rest: float = -55.0
    v_reset: float = -65.0
    v_threshold: float = -48.0
    tau_m: float = 10.0
    tau_adapt: float = 150.0
    a_adapt: float = 0.05
    gain: float = 2.0
    serotonin: float = 0.0
    dt: float = 0.5

    @classmethod
    def with_serotonin(cls, level: float) -> LugaroCell:
        return cls(serotonin=max(0.0, min(1.0, level)))

    def step(self, current: float = 0.0) -> int:
        effective_gain = self.gain * (1.0 + 0.5 * self.serotonin)
        inp = effective_gain * current
        dv = (-(self.v - self.v_rest) - self.adapt + inp) / self.tau_m
        self.v += self.dt * dv
        da = (self.a_adapt * (self.v - self.v_rest) - self.adapt) / self.tau_adapt
        self.adapt += self.dt * da

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.adapt += 1.0
            return 1

        self.v = max(-100.0, min(60.0, self.v))
        if not math.isfinite(self.v):
            self.v = self.v_reset
        if not math.isfinite(self.adapt):
            self.adapt = 0.0
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.adapt = 0.0
