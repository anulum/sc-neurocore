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

    def __post_init__(self) -> None:
        self._validate_state()

    @classmethod
    def with_serotonin(cls, level: float) -> LugaroCell:
        return cls(serotonin=max(0.0, min(1.0, level)))

    def step(self, current: float = 0.0) -> int:
        self._validate_state()
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        effective_gain = self.gain * (1.0 + 0.5 * self.serotonin)
        inp = effective_gain * current
        v_inf = self.v_rest + inp - self.adapt
        v_next = v_inf + (self.v - v_inf) * math.exp(-self.dt / self.tau_m)
        adapt_inf = max(0.0, self.a_adapt * max(0.0, v_next - self.v_rest))
        adapt_next = adapt_inf + (self.adapt - adapt_inf) * math.exp(-self.dt / self.tau_adapt)
        adapt_next = max(0.0, adapt_next)

        if not math.isfinite(v_next) or not math.isfinite(adapt_next):
            raise ValueError("lugaro cell integration produced non-finite state")

        if v_next >= self.v_threshold:
            self.v = self.v_reset
            self.adapt = adapt_next + 1.0
            return 1

        self.v = max(-100.0, min(60.0, v_next))
        self.adapt = adapt_next
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.adapt = 0.0

    def _validate_state(self) -> None:
        finite_values = (
            self.v,
            self.adapt,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau_m,
            self.tau_adapt,
            self.a_adapt,
            self.gain,
            self.serotonin,
            self.dt,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("lugaro cell state and parameters must be finite")
        if self.tau_m <= 0.0 or self.tau_adapt <= 0.0 or self.dt <= 0.0:
            raise ValueError("lugaro cell time constants and timestep must be positive")
        if self.a_adapt < 0.0:
            raise ValueError("lugaro cell adaptation coupling must be non-negative")
        if self.gain < 0.0:
            raise ValueError("lugaro cell gain must be non-negative")
        if not -100.0 <= self.v <= 60.0:
            raise ValueError("lugaro cell membrane potential must stay in [-100, 60] mV")
        if not 0.0 <= self.serotonin <= 1.0:
            raise ValueError("lugaro cell serotonin must stay in [0, 1]")
        if self.adapt < 0.0:
            raise ValueError("lugaro cell adaptation current must be non-negative")
        if self.v_threshold <= self.v_reset or self.v_threshold <= self.v_rest:
            raise ValueError("lugaro cell threshold must exceed reset and rest potentials")
