# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benda & Herz 2003 universal adaptation model

"""Source-faithful deterministic Benda–Herz rate adaptation.

This specialisation implements equations (8) and (45) from Benda & Herz
(2003): a universal adapting rate model coupled to their simplest phase spike
generator. Frequencies are expressed in hertz and time in milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BendaHerzNeuron:
    """Benda–Herz universal rate adaptation with deterministic phase spikes.

    The chosen paper example is ``f0(x) = onset_gain * sqrt(max(x-rheobase, 0))``
    with ``gamma(f)=0`` and ``A_inf(f)=adaptation_slope*f``. The phase follows
    ``dphase/dt=f/1000`` and resets exactly to zero at threshold one.

    Reference: Benda, J. & Herz, A. V. M. (2003), Neural Computation 15,
    2523–2564, DOI 10.1162/089976603322385063.
    """

    a: float = 0.0
    phase: float = 0.0
    onset_gain: float = 60.0
    rheobase: float = 0.0
    adaptation_slope: float = 0.1
    tau_a: float = 100.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.a) or self.a < 0.0:
            raise ValueError("a must be finite and non-negative")
        if not math.isfinite(self.phase) or not 0.0 <= self.phase < 1.0:
            raise ValueError("phase must be finite and within [0, 1)")
        for name in ("onset_gain", "tau_a", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.rheobase):
            raise ValueError("rheobase must be finite")
        if not math.isfinite(self.adaptation_slope) or self.adaptation_slope < 0.0:
            raise ValueError("adaptation_slope must be finite and non-negative")

    def _f_onset(self, effective_current: float) -> float:
        """Return the paper's square-root onset curve in hertz."""
        if not math.isfinite(effective_current):
            raise ValueError("effective current must be finite")
        drive = max(effective_current - self.rheobase, 0.0)
        rate = self.onset_gain * math.sqrt(drive)
        if not math.isfinite(rate):
            raise ValueError("onset rate must be finite")
        return rate

    def _rhs(self, adaptation: float, current: float) -> tuple[float, float]:
        if not math.isfinite(adaptation) or adaptation < 0.0:
            raise ValueError("adaptation RK4 stage must be finite and non-negative")
        rate = self._f_onset(current - adaptation)
        da = (self.adaptation_slope * rate - adaptation) / self.tau_a
        return da, rate / 1000.0

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        """Return candidate adaptation, phase, and RK4-averaged rate."""
        k1a, k1p = self._rhs(self.a, current)
        k2a, k2p = self._rhs(self.a + 0.5 * self.dt * k1a, current)
        k3a, k3p = self._rhs(self.a + 0.5 * self.dt * k2a, current)
        k4a, k4p = self._rhs(self.a + self.dt * k3a, current)
        scale = self.dt / 6.0
        next_a = self.a + scale * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        phase_increment = scale * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        next_phase = self.phase + phase_increment
        rate = phase_increment * 1000.0 / self.dt
        if not math.isfinite(next_a) or next_a < 0.0:
            raise ValueError("adaptation RK4 candidate must be finite and non-negative")
        if not math.isfinite(next_phase) or next_phase < 0.0 or next_phase >= 2.0:
            raise ValueError("phase candidate must be finite and permit at most one spike")
        return next_a, next_phase, rate

    def step(self, current: float) -> int:
        """Advance one sample and emit the paper's deterministic phase spike."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        next_a, next_phase, _ = self._rk4_candidate(current)
        spike = next_phase >= 1.0
        self.a = next_a
        self.phase = 0.0 if spike else next_phase
        return int(spike)

    def reset(self) -> None:
        """Restore the paper state variables to their initial values."""
        self.a = 0.0
        self.phase = 0.0
