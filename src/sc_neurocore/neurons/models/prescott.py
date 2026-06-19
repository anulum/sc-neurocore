# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Prescott 2008 — Type I/II/III excitability via M-current

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PrescottNeuron:
    """Prescott 2008 two-state excitability model with M-current tuning.

    Reference: Prescott, S.A. et al. (2008). PLoS Comput. Biol. 4:e1000198.
    """

    v: float = -65.0
    w: float = 0.0
    g_fast: float = 20.0
    g_slow: float = 20.0
    g_l: float = 2.0
    e_fast: float = 50.0
    e_slow: float = -100.0
    e_l: float = -70.0
    beta_w: float = -21.0
    gamma_w: float = 15.0
    tau_w: float = 100.0
    phi: float = 0.15
    dt: float = 0.1
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self._validate_configuration(coerce=True)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0.0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _validate_recovery(value: float) -> float:
        recovery = float(value)
        if not math.isfinite(recovery):
            raise FloatingPointError("Prescott w state must be finite")
        if not 0.0 <= recovery <= 1.0:
            raise FloatingPointError("Prescott w state must remain in [0, 1]")
        return recovery

    @classmethod
    def _validate_state(cls, v: float, w: float) -> tuple[float, float]:
        voltage = float(v)
        if not math.isfinite(voltage):
            raise FloatingPointError("Prescott voltage state must be finite")
        return voltage, cls._validate_recovery(w)

    def _validate_configuration(self, *, coerce: bool = False) -> None:
        for name in (
            "v",
            "w",
            "g_fast",
            "g_slow",
            "g_l",
            "e_fast",
            "e_slow",
            "e_l",
            "beta_w",
            "gamma_w",
            "tau_w",
            "phi",
            "dt",
            "v_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if coerce:
                setattr(self, name, value)
        if self.g_fast < 0.0 or self.g_slow < 0.0 or self.g_l < 0.0:
            raise ValueError("conductances must be non-negative")
        if self.gamma_w <= 0.0:
            raise ValueError("gamma_w must be positive")
        if self.tau_w <= 0.0:
            raise ValueError("tau_w must be positive")
        if self.phi < 0.0:
            raise ValueError("phi must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self._validate_state(self.v, self.w)

    def _derivatives(self, v: float, w: float, current: float) -> tuple[float, float]:
        voltage, recovery = self._validate_state(v, w)
        m_inf = self._sigmoid((voltage + 20.0) / 15.0)
        w_inf = self._sigmoid((voltage - self.beta_w) / self.gamma_w)
        i_fast = self.g_fast * m_inf * (voltage - self.e_fast)
        i_slow = self.g_slow * recovery * (voltage - self.e_slow)
        i_l = self.g_l * (voltage - self.e_l)
        dv = -i_fast - i_slow - i_l + current
        dw = self.phi * (w_inf - recovery) / self.tau_w
        if not math.isfinite(dv) or not math.isfinite(dw):
            raise FloatingPointError("Prescott derivative must remain finite")
        return dv, dw

    def _rk4_step(self, current: float) -> tuple[float, float]:
        v0, w0 = self._validate_state(self.v, self.w)
        dt = self.dt
        k1_v, k1_w = self._derivatives(v0, w0, current)
        k2_v, k2_w = self._derivatives(v0 + 0.5 * dt * k1_v, w0 + 0.5 * dt * k1_w, current)
        k3_v, k3_w = self._derivatives(v0 + 0.5 * dt * k2_v, w0 + 0.5 * dt * k2_w, current)
        k4_v, k4_w = self._derivatives(v0 + dt * k3_v, w0 + dt * k3_w, current)
        return self._validate_state(
            v0 + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0,
            w0 + dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0,
        )

    def step(self, current: float) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        self._validate_configuration()
        v_prev = self.v
        next_v, next_w = self._rk4_step(drive)
        self.v, self.w = next_v, next_w
        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.w = 0.0
