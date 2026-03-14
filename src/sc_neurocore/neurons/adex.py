# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AdExNeuron:
    """Adaptive Exponential Integrate-and-Fire. Brette & Gerstner 2005.

    dv/dt = -(v - v_rest)/tau + delta_T * exp((v - v_rh)/delta_T) / tau - w/C + I/C
    dw/dt = (a * (v - v_rest) - w) / tau_w
    if v >= v_threshold: v = v_reset, w += b
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    tau_w: float = 100.0
    a: float = 0.5
    b: float = 7.0
    c_m: float = 200.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        exp_term = self.delta_t * np.exp(np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0))
        dv = (-(self.v - self.v_rest) + exp_term - self.w + current) / self.tau * self.dt
        dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt

        self.v += dv
        self.w += dw

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.w += self.b
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.w = 0.0


@dataclass
class ExpIFNeuron:
    """Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003."""

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        exp_term = self.delta_t * np.exp(np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0))
        dv = (-(self.v - self.v_rest) + exp_term + current) / self.tau * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest


@dataclass
class LapicqueNeuron:
    """Lapicque 1907 — classical RC integrate-and-fire.

    tau * dv/dt = -(v - v_rest) + R * I
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau: float = 20.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        dv = (-(self.v - self.v_rest) + self.resistance * current) / self.tau * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest


@dataclass
class AlphaNeuron:
    """Alpha-synapse neuron. Rall 1967.

    Dual excitatory/inhibitory synaptic currents with alpha-function kinetics.
    """

    v: float = 0.0
    i_exc: float = 0.0
    i_inh: float = 0.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    tau_v: float = 20.0
    tau_exc: float = 5.0
    tau_inh: float = 10.0
    dt: float = 1.0

    def step(self, exc_current: float, inh_current: float = 0.0) -> int:
        self.i_exc += (-self.i_exc / self.tau_exc + exc_current) * self.dt
        self.i_inh += (-self.i_inh / self.tau_inh + inh_current) * self.dt
        dv = (-(self.v - self.v_rest) + self.i_exc - self.i_inh) / self.tau_v * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_rest
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.i_exc = 0.0
        self.i_inh = 0.0
