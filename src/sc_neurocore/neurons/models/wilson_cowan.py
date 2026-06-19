# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-Cowan 1972 — excitatory/inhibitory population

from __future__ import annotations

import math
from dataclasses import dataclass

_STATE_NAMES = ("e", "i")
_PARAM_NAMES = ("w_ee", "w_ei", "w_ie", "w_ii", "tau_e", "tau_i", "a", "theta", "dt")
_NON_NEGATIVE_PARAMS = ("w_ee", "w_ei", "w_ie", "w_ii")
_STRICTLY_POSITIVE_PARAMS = ("tau_e", "tau_i", "a", "dt")


@dataclass
class WilsonCowanUnit:
    """Wilson-Cowan 1972 — excitatory/inhibitory population rate model.

    τ_e dE/dt = -E + S(w_ee·E - w_ei·I + I_ext)
    τ_i dI/dt = -I + S(w_ie·E - w_ii·I)
    S(x) = 1/(1 + exp(-a(x-θ))) - 1/(1 + exp(aθ))

    Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.
    """

    e: float = 0.1
    i: float = 0.05
    w_ee: float = 10.0
    w_ei: float = 6.0
    w_ie: float = 10.0
    w_ii: float = 1.0
    tau_e: float = 1.0
    tau_i: float = 2.0
    a: float = 1.2
    theta: float = 4.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        self._validate_configuration(coerce=True)

    def _validate_configuration(self, *, coerce: bool = False) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if coerce:
                setattr(self, name, value)
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        self._validate_state(self.e, self.i)

    @staticmethod
    def _logistic(z: float) -> float:
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def _validate_rate(self, name: str, value: float) -> float:
        rate = float(value)
        baseline = self._logistic(-self.a * self.theta)
        lower = -baseline
        upper = 1.0 - baseline
        if not math.isfinite(rate) or rate < lower or rate > upper:
            raise FloatingPointError(f"{name} rate must remain in Wilson-Cowan sigmoid range")
        return rate

    def _validate_state(self, e: float, i: float) -> tuple[float, float]:
        return self._validate_rate("e", e), self._validate_rate("i", i)

    def _sigmoid(self, x: float) -> float:
        # Published Wilson-Cowan 1972 two-term sigmoid:
        #   S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
        # The subtracted baseline makes S(0) = 0 exactly. The earlier
        # one-term implementation left an artefactual bias
        # `1/(1+exp(aθ))` ≈ 0.008 at x = 0 that shifted the model's
        # fixed points and suppressed the Hopf oscillator regime.
        # math.exp on scalars keeps the per-step cost ~0.7 µs.
        drive = float(x)
        if not math.isfinite(drive):
            raise ValueError("sigmoid input must be finite")
        return self._logistic(self.a * (drive - self.theta)) - self._logistic(-self.a * self.theta)

    def _derivatives(self, e: float, i: float, drive: float) -> tuple[float, float]:
        se = self._sigmoid(self.w_ee * e - self.w_ei * i + drive)
        si = self._sigmoid(self.w_ie * e - self.w_ii * i)
        de = (-e + se) / self.tau_e
        di = (-i + si) / self.tau_i
        if not math.isfinite(de) or not math.isfinite(di):
            raise FloatingPointError("Wilson-Cowan derivative must remain finite")
        return de, di

    def step(self, ext_input: float = 0.0) -> float:
        drive = float(ext_input)
        if not math.isfinite(drive):
            raise ValueError("external input must be finite")

        self._validate_configuration()
        e, i = self._validate_state(self.e, self.i)

        k1_e, k1_i = self._derivatives(e, i, drive)
        k2_e, k2_i = self._derivatives(
            e + 0.5 * self.dt * k1_e,
            i + 0.5 * self.dt * k1_i,
            drive,
        )
        k3_e, k3_i = self._derivatives(
            e + 0.5 * self.dt * k2_e,
            i + 0.5 * self.dt * k2_i,
            drive,
        )
        k4_e, k4_i = self._derivatives(
            e + self.dt * k3_e,
            i + self.dt * k3_i,
            drive,
        )
        next_e = e + self.dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0
        next_i = i + self.dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0
        self.e, self.i = self._validate_state(next_e, next_i)
        return self.e

    def reset(self) -> None:
        self.e, self.i = 0.1, 0.05
