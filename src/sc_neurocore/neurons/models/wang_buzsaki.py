# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wang-Buzsáki 1996 — fast-spiking GABAergic interneuron

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class WangBuzsakiNeuron:
    """Wang-Buzsáki 1996 — fast-spiking GABAergic interneuron.

    3 ODEs. Simplified HH with only Na + K delayed rectifier.
    Designed for gamma (30-80 Hz) oscillation modelling.

    Reference: Wang, X.-J. & Buzsáki, G. (1996). J. Neurosci. 16:6402–6413.
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    g_na: float = 35.0
    g_k: float = 9.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        for name in (
            "v",
            "h",
            "n",
            "g_na",
            "g_k",
            "g_l",
            "e_na",
            "e_k",
            "e_l",
            "c_m",
            "phi",
            "dt",
            "v_threshold",
        ):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, float(value))
        for name in ("g_na", "g_k", "g_l", "c_m", "phi", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @staticmethod
    def _safe_exp(arg: float, label: str) -> float:
        try:
            out = math.exp(arg)
        except OverflowError as exc:
            raise FloatingPointError(f"Wang-Buzsaki {label} rate overflowed") from exc
        if not math.isfinite(out):
            raise FloatingPointError(f"Wang-Buzsaki {label} rate became non-finite")
        return out

    @classmethod
    def _gating_rates(cls, v: float) -> tuple[float, float, float, float, float]:
        if not math.isfinite(v):
            raise FloatingPointError("Wang-Buzsaki voltage became non-finite")
        alpha_m = (
            0.1 * (v + 35.0) / (1.0 - cls._safe_exp(-(v + 35.0) / 10.0, "m-alpha"))
            if abs(v + 35.0) > 1e-6
            else 1.0
        )
        beta_m = 4.0 * cls._safe_exp(-(v + 60.0) / 18.0, "m-beta")
        denom_m = alpha_m + beta_m
        if denom_m == 0.0 or not math.isfinite(denom_m):
            raise FloatingPointError("Wang-Buzsaki sodium activation denominator became invalid")
        m_inf = alpha_m / denom_m
        alpha_h = 0.07 * cls._safe_exp(-(v + 58.0) / 20.0, "h-alpha")
        beta_h = 1.0 / (1.0 + cls._safe_exp(-(v + 28.0) / 10.0, "h-beta"))
        alpha_n = (
            0.01 * (v + 34.0) / (1.0 - cls._safe_exp(-(v + 34.0) / 10.0, "n-alpha"))
            if abs(v + 34.0) > 1e-6
            else 0.1
        )
        beta_n = 0.125 * cls._safe_exp(-(v + 44.0) / 80.0, "n-beta")
        rates = (m_inf, alpha_h, beta_h, alpha_n, beta_n)
        if not all(math.isfinite(value) for value in rates):
            raise FloatingPointError("Wang-Buzsaki gating rate became non-finite")
        return rates

    @staticmethod
    def _validate_state(v: float, h: float, n: float) -> tuple[float, float, float]:
        if not (math.isfinite(v) and math.isfinite(h) and math.isfinite(n)):
            raise FloatingPointError("Wang-Buzsaki state became non-finite")
        return float(v), float(h), float(n)

    def step(self, current: float) -> int:
        if not isinstance(current, int | float) or not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        current = float(current)
        v_prev = self.v
        v, h, n = self._validate_state(self.v, self.h, self.n)
        for _ in range(int(0.5 / max(self.dt, 0.001))):
            # m is instantaneous (m_inf)
            m_inf, alpha_h, beta_h, alpha_n, beta_n = self._gating_rates(v)

            next_h = h + self.phi * (alpha_h * (1 - h) - beta_h * h) * self.dt
            next_n = n + self.phi * (alpha_n * (1 - n) - beta_n * n) * self.dt

            i_na = self.g_na * m_inf**3 * next_h * (v - self.e_na)
            i_k = self.g_k * next_n**4 * (v - self.e_k)
            i_l = self.g_l * (v - self.e_l)

            next_v = v + (-i_na - i_k - i_l + current) / self.c_m * self.dt
            v, h, n = self._validate_state(next_v, next_h, next_n)

        self.v, self.h, self.n = v, h, n
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h, self.n = 0.8, 0.1
