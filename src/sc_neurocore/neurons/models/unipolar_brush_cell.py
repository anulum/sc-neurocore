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

    def __post_init__(self) -> None:
        self._validate_configuration()
        self._validate_state()

    @staticmethod
    def _first_order_relaxation(
        previous: float, steady_state: float, dt: float, tau: float
    ) -> float:
        return previous + (steady_state - previous) * (-math.expm1(-dt / tau))

    def _validate_configuration(self) -> None:
        for name in (
            "v_rest",
            "v_reset",
            "v_threshold",
            "tau_m",
            "tau_persistent",
            "persistent_gain",
            "gain",
            "dt",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.tau_m <= 0.0:
            raise ValueError("tau_m must be positive")
        if self.tau_persistent <= 0.0:
            raise ValueError("tau_persistent must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.persistent_gain < 0.0:
            raise ValueError("persistent_gain must be non-negative")
        if self.gain < 0.0:
            raise ValueError("gain must be non-negative")
        if self.v_reset >= self.v_threshold:
            raise ValueError("v_reset must be below v_threshold")

    def _validate_state(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("v state must be finite")
        if not -100.0 <= self.v <= 60.0:
            raise ValueError("v state must remain within [-100, 60] mV")
        if not math.isfinite(self.persistent):
            raise ValueError("persistent state must be finite")
        if self.persistent < 0.0:
            raise ValueError("persistent state must be non-negative")

    def step(self, current: float = 0.0) -> int:
        self._validate_configuration()
        self._validate_state()
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        inp = self.gain * max(0.0, current)
        if not math.isfinite(inp):
            raise ValueError("input drive must be finite")

        next_persistent = self._first_order_relaxation(
            self.persistent,
            self.persistent_gain * inp,
            self.dt,
            self.tau_persistent,
        )
        next_persistent = max(0.0, next_persistent)
        voltage_steady_state = self.v_rest + inp + next_persistent
        next_v = self._first_order_relaxation(self.v, voltage_steady_state, self.dt, self.tau_m)

        if not math.isfinite(next_persistent) or not math.isfinite(next_v):
            raise ValueError("candidate state must be finite")

        self.persistent = next_persistent
        if next_v >= self.v_threshold:
            self.v = self.v_reset
            return 1

        self.v = max(-100.0, min(60.0, next_v))
        return 0

    def reset(self) -> None:
        self._validate_configuration()
        self.v = self.v_rest
        self.persistent = 0.0
