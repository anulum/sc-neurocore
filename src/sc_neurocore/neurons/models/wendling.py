# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wendling et al. 2002 — extended Jansen-Rit with slow

from __future__ import annotations

import math
from dataclasses import dataclass

_STATE_NAMES = ("y0", "y5", "y1", "y6", "y2", "y7", "y3", "y8", "y4", "y9")
_PARAM_NAMES = (
    "a_exc",
    "b_fast",
    "g_slow",
    "a_rate",
    "b_rate",
    "g_rate",
    "c",
    "e0",
    "v0",
    "r",
    "dt",
)
_STRICTLY_POSITIVE_PARAMS = (
    "a_exc",
    "b_fast",
    "g_slow",
    "a_rate",
    "b_rate",
    "g_rate",
    "e0",
    "r",
    "dt",
)


@dataclass
class WendlingNeuron:
    """Wendling et al. 2002 — extended Jansen-Rit with slow GABA_B inhibition.

    10 ODEs: 4 populations (pyramidal, excitatory, fast inhibitory, slow
    inhibitory) x 2 states each + 2 for slow inhibitory PSP.
    Reproduces epileptiform EEG patterns.

    Reference: Wendling, F. et al. (2002). Biol. Cybern. 86:97–108.
    """

    y0: float = 0.0
    y5: float = 0.0
    y1: float = 0.0
    y6: float = 0.0
    y2: float = 0.0
    y7: float = 0.0
    y3: float = 0.0
    y8: float = 0.0
    y4: float = 0.0
    y9: float = 0.0

    a_exc: float = 3.25
    b_fast: float = 22.0
    g_slow: float = 10.0
    a_rate: float = 100.0
    b_rate: float = 500.0
    g_rate: float = 20.0
    c: float = 135.0
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    dt: float = 0.001

    def __post_init__(self) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.c < 0.0:
            raise ValueError("c must be non-negative")

    @staticmethod
    def _require_finite(name: str, value: float) -> float:
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"{name} must be finite")
        return out

    def _validate_state(self, values: tuple[float, ...] | None = None) -> tuple[float, ...]:
        state = (
            values if values is not None else tuple(getattr(self, name) for name in _STATE_NAMES)
        )
        checked = tuple(
            self._require_finite(name, value) for name, value in zip(_STATE_NAMES, state)
        )
        if len(checked) != len(_STATE_NAMES):
            raise ValueError("Wendling state vector has invalid dimension")
        return checked

    def _sigmoid(self, x: float) -> float:
        drive = self._require_finite("sigmoid input", x)
        exponent = self.r * (self.v0 - drive)
        if exponent >= 0.0:
            exp_neg = math.exp(-exponent)
            return 2.0 * self.e0 * exp_neg / (1.0 + exp_neg)
        return 2.0 * self.e0 / (1.0 + math.exp(exponent))

    def step(self, p_ext: float = 220.0) -> float:
        p_ext = self._require_finite("p_ext", p_ext)
        y0, y5, y1, y6, y2, y7, y3, y8, y4, y9 = self._validate_state()

        sig_1_2_3_4 = self._sigmoid(y1 - y2 - y3)
        sig_0 = self._sigmoid(self.c * 0.8 * y0)
        sig_fast = self._sigmoid(self.c * 0.25 * y0)
        sig_slow = self._sigmoid(self.c * 0.1 * y0)

        dy0 = y5
        dy5 = self.a_exc * self.a_rate * sig_1_2_3_4 - 2 * self.a_rate * y5 - self.a_rate**2 * y0
        dy1 = y6
        dy6 = (
            self.a_exc * self.a_rate * (p_ext + self.c * 0.8 * sig_0)
            - 2 * self.a_rate * y6
            - self.a_rate**2 * y1
        )
        dy2 = y7
        dy7 = (
            self.b_fast * self.b_rate * self.c * 0.25 * sig_fast
            - 2 * self.b_rate * y7
            - self.b_rate**2 * y2
        )
        dy3 = y8
        dy8 = (
            self.g_slow * self.g_rate * self.c * 0.1 * sig_slow
            - 2 * self.g_rate * y8
            - self.g_rate**2 * y3
        )

        candidate = (
            y0 + dy0 * self.dt,
            y5 + dy5 * self.dt,
            y1 + dy1 * self.dt,
            y6 + dy6 * self.dt,
            y2 + dy2 * self.dt,
            y7 + dy7 * self.dt,
            y3 + dy3 * self.dt,
            y8 + dy8 * self.dt,
            y4,
            y9,
        )
        (
            self.y0,
            self.y5,
            self.y1,
            self.y6,
            self.y2,
            self.y7,
            self.y3,
            self.y8,
            self.y4,
            self.y9,
        ) = self._validate_state(candidate)

        return self.y1 - self.y2 - self.y3

    def reset(self) -> None:
        self.y0 = self.y1 = self.y2 = self.y3 = self.y4 = 0.0
        self.y5 = self.y6 = self.y7 = self.y8 = self.y9 = 0.0
