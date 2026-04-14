# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cerebellar Unipolar Brush Cell

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class UnipolarBrushCell:
    """Unipolar brush cell (UBC) — excitatory vestibular cerebellum interneuron.

    LIF with slow NMDA-like persistent current that prolongs mossy fibre
    bursts into sustained granule cell activation. Giant 1:1 synapse.

    Reference: Bhatt et al. (1994) J Comp Neurol 349:560;
    Diana et al. (2007) J Neurosci 27:4374.
    """

    v: float = -65.0
    persistent: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 8.0
    tau_persistent: float = 200.0
    persistent_gain: float = 0.5
    gain: float = 2.5
    dt: float = 0.5

    def step(self, current: float = 0.0) -> int:
        inp = self.gain * max(0.0, current)
        dp = (self.persistent_gain * inp - self.persistent) / self.tau_persistent
        self.persistent += self.dt * dp
        self.persistent = max(0.0, self.persistent)

        dv = (-(self.v - self.v_rest) + inp + self.persistent) / self.tau_m
        self.v += self.dt * dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1

        self.v = max(-100.0, min(60.0, self.v))
        if not math.isfinite(self.v):
            self.v = self.v_reset
        if not math.isfinite(self.persistent):
            self.persistent = 0.0
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.persistent = 0.0
