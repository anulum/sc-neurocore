# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class InhibitoryLIFNeuron:
    """Inhibitory LIF — 2025, temporal inhibitory mechanism.

    After spiking, a decaying inhibitory trace suppresses the membrane
    for a learned duration, shaping temporal coding.
    """

    v: float = 0.0
    inh_trace: float = 0.0
    tau_m: float = 10.0
    tau_inh: float = 5.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    inh_strength: float = 0.5
    dt: float = 1.0
    alpha_m: float = field(init=False)
    alpha_inh: float = field(init=False)

    def __post_init__(self):
        self.alpha_m = np.exp(-self.dt / self.tau_m)
        self.alpha_inh = np.exp(-self.dt / self.tau_inh)

    def step(self, current: float) -> int:
        self.inh_trace *= self.alpha_inh
        self.v = self.alpha_m * self.v + current - self.inh_strength * self.inh_trace
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.inh_trace += 1.0
            return 1
        return 0

    def reset(self):
        self.v, self.inh_trace = 0.0, 0.0
