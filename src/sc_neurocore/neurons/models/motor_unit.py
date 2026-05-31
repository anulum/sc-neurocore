# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Motor Unit (Alpha Motor Neuron + Muscle Fibre)

from __future__ import annotations

import math
from dataclasses import dataclass


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _relax(previous: float, steady: float, tau: float, dt: float) -> float | None:
    if not _all_finite(previous, steady, tau, dt) or tau <= 0.0 or dt <= 0.0:
        return None
    return steady + (previous - steady) * math.exp(-dt / tau)


def _valid_voltage(value: float) -> bool:
    return math.isfinite(value) and -150.0 <= value <= 100.0


def _valid_force(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


@dataclass
class MotorUnit:
    """Motor unit — alpha motor neuron + muscle fibre.

    Each spike triggers a muscle twitch. Force output is summation of
    overlapping twitches (rate coding). Twitch modelled as critically-
    damped second-order: f(t) = A · (t/τ) · exp(1 - t/τ).

    Reference: Fuglevand et al. (1993) J Neurophysiol 70(6);
    Heckman & Enoka (2012) Compr Physiol 2(4).
    """

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    adapt: float = 0.0
    tau_adapt: float = 100.0
    a_adapt: float = 0.2
    gain: float = 1.0
    force: float = 0.0
    twitch_amp: float = 0.05
    tau_twitch: float = 90.0
    force_decay: float = 0.0
    dt: float = 0.5

    def _valid_state(self) -> bool:
        return (
            _valid_voltage(self.v)
            and _valid_voltage(self.v_rest)
            and _valid_voltage(self.v_reset)
            and _valid_voltage(self.v_threshold)
            and _valid_force(self.force)
            and _all_finite(
                self.tau_m,
                self.adapt,
                self.tau_adapt,
                self.a_adapt,
                self.gain,
                self.twitch_amp,
                self.tau_twitch,
                self.force_decay,
                self.dt,
            )
            and self.tau_m > 0.0
            and self.tau_adapt > 0.0
            and self.tau_twitch > 0.0
            and self.dt > 0.0
            and self.gain >= 0.0
            and self.twitch_amp >= 0.0
            and self.v_reset < self.v_threshold
        )

    @classmethod
    def slow(cls) -> MotorUnit:
        """Slow motor unit (type S): small, fatigue-resistant, low force."""
        return cls()

    @classmethod
    def fast(cls) -> MotorUnit:
        """Fast motor unit (type FF): large, fatigable, high force."""
        return cls(
            tau_m=6.0,
            tau_adapt=50.0,
            a_adapt=0.1,
            twitch_amp=0.3,
            tau_twitch=30.0,
        )

    def step(self, drive: float = 0.0) -> int:
        if not math.isfinite(drive) or not self._valid_state():
            return 0

        force = self.force * math.exp(-self.dt / self.tau_twitch)
        input_drive = self.gain * max(0.0, drive) - self.adapt
        v_target = self.v_rest + input_drive
        v_candidate = _relax(self.v, v_target, self.tau_m, self.dt)
        if v_candidate is None or not _valid_voltage(v_candidate):
            return 0

        adapt_target = self.a_adapt * (v_candidate - self.v_rest)
        adapt_candidate = _relax(self.adapt, adapt_target, self.tau_adapt, self.dt)
        if adapt_candidate is None or not math.isfinite(adapt_candidate):
            return 0

        spike = 0
        if v_candidate >= self.v_threshold:
            v_candidate = self.v_reset
            force = min(1.0, force + self.twitch_amp)
            spike = 1

        if not (_valid_voltage(v_candidate) and _valid_force(force)):
            return 0

        self.v = v_candidate
        self.adapt = adapt_candidate
        self.force = force
        if spike:
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.adapt = 0.0
        self.force = 0.0
