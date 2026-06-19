# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Traub & Miles 1991 — reduced hippocampal CA3 pyramidal

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TraubMilesNeuron:
    """Traub & Miles 1991 — reduced hippocampal CA3 pyramidal.

    Reference: Traub, R.D. & Miles, R. (1991). Neuronal Networks of the Hippocampus. Cambridge Univ. Press.
    """

    v: float = -67.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    g_na: float = 100.0
    g_k: float = 80.0
    g_l: float = 0.1
    e_na: float = 50.0
    e_k: float = -100.0
    e_l: float = -67.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self._validate_configuration(coerce=True)

    def _validate_configuration(self, *, coerce: bool = False) -> None:
        for name in (
            "v",
            "m",
            "h",
            "n",
            "g_na",
            "g_k",
            "g_l",
            "e_na",
            "e_k",
            "e_l",
            "dt",
            "v_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if coerce:
                setattr(self, name, value)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        for name in ("g_na", "g_k", "g_l"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self._validate_state(self.v, self.m, self.h, self.n)

    @staticmethod
    def _validate_gate(name: str, value: float) -> float:
        gate = float(value)
        if not math.isfinite(gate) or gate < 0.0 or gate > 1.0:
            raise FloatingPointError(f"{name} gate must remain in [0, 1]")
        return gate

    @classmethod
    def _validate_state(
        cls, v: float, m: float, h: float, n: float
    ) -> tuple[float, float, float, float]:
        voltage = float(v)
        if not math.isfinite(voltage):
            raise FloatingPointError("Traub-Miles voltage state must be finite")
        return (
            voltage,
            cls._validate_gate("m", m),
            cls._validate_gate("h", h),
            cls._validate_gate("n", n),
        )

    @staticmethod
    def _rates(v: float) -> tuple[float, float, float, float, float, float]:
        try:
            d = v + 54.0
            am = 0.32 * d / (1.0 - math.exp(-d / 4.0)) if abs(d) > 1e-6 else 8.0
            d2 = v + 27.0
            bm = 0.28 * d2 / (math.exp(d2 / 5.0) - 1.0) if abs(d2) > 1e-6 else 5.6
            ah = 0.128 * math.exp(-(v + 50.0) / 18.0)
            bh = 4.0 / (1.0 + math.exp(-(v + 27.0) / 5.0))
            d3 = v + 52.0
            an = 0.032 * d3 / (1.0 - math.exp(-d3 / 5.0)) if abs(d3) > 1e-6 else 0.32
            bn = 0.5 * math.exp(-(v + 57.0) / 40.0)
        except (OverflowError, ZeroDivisionError) as exc:
            raise FloatingPointError("Traub-Miles rate evaluation overflowed") from exc
        rates = (am, bm, ah, bh, an, bn)
        if not all(math.isfinite(rate) and rate >= 0.0 for rate in rates):
            raise FloatingPointError("Traub-Miles rates must be finite and non-negative")
        return rates

    def _derivatives(
        self, v: float, m: float, h: float, n: float, drive: float
    ) -> tuple[float, float, float, float]:
        v, m, h, n = self._validate_state(v, m, h, n)
        am, bm, ah, bh, an, bn = self._rates(v)
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n
        i_na = self.g_na * m**3 * h * (v - self.e_na)
        i_k = self.g_k * n**4 * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        dv = -i_na - i_k - i_l + drive
        values = (dv, dm, dh, dn, i_na, i_k, i_l)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Traub-Miles derivative must remain finite")
        return dv, dm, dh, dn

    def step(self, current: float) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")

        self._validate_configuration()
        v, m, h, n = self._validate_state(self.v, self.m, self.h, self.n)
        v_prev = v
        for _ in range(10):
            k1_v, k1_m, k1_h, k1_n = self._derivatives(v, m, h, n, drive)
            k2_v, k2_m, k2_h, k2_n = self._derivatives(
                v + 0.5 * self.dt * k1_v,
                m + 0.5 * self.dt * k1_m,
                h + 0.5 * self.dt * k1_h,
                n + 0.5 * self.dt * k1_n,
                drive,
            )
            k3_v, k3_m, k3_h, k3_n = self._derivatives(
                v + 0.5 * self.dt * k2_v,
                m + 0.5 * self.dt * k2_m,
                h + 0.5 * self.dt * k2_h,
                n + 0.5 * self.dt * k2_n,
                drive,
            )
            k4_v, k4_m, k4_h, k4_n = self._derivatives(
                v + self.dt * k3_v,
                m + self.dt * k3_m,
                h + self.dt * k3_h,
                n + self.dt * k3_n,
                drive,
            )
            next_v = v + self.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
            next_m = m + self.dt * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m) / 6.0
            next_h = h + self.dt * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h) / 6.0
            next_n = n + self.dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0
            v, m, h, n = self._validate_state(next_v, next_m, next_h, next_n)
        self.v, self.m, self.h, self.n = v, m, h, n
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.m, self.h, self.n = -67.0, 0.05, 0.6, 0.3
