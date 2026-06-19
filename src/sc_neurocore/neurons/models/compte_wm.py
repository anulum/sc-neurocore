# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compte et al. 2000 — NMDA-based working memory neuron

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CompteWMNeuron:
    """Compte et al. NMDA-based working-memory neuron with Mg2+ block."""

    v: float = -70.0
    s_ampa: float = 0.0
    s_nmda: float = 0.0
    x_nmda: float = 0.0
    s_gaba: float = 0.0
    g_l: float = 0.025
    g_ampa: float = 0.005
    g_nmda: float = 0.165
    g_gaba: float = 0.013
    e_l: float = -70.0
    e_exc: float = 0.0
    e_inh: float = -70.0
    c_m: float = 0.5
    mg: float = 1.0
    tau_ampa: float = 2.0
    tau_nmda: float = 100.0
    tau_x: float = 2.0
    alpha_nmda: float = 0.5
    v_threshold: float = -50.0
    v_reset: float = -55.0
    dt: float = 0.1

    _V_MIN: float = -200.0
    _V_MAX: float = 100.0
    _GATE_MAX: float = 1.0e6
    _GABA_TAU: float = 5.0

    def __post_init__(self) -> None:
        self._validated_state()

    @staticmethod
    def _finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _decay(cls, dt: float, tau: float, name: str) -> float:
        ratio = -dt / tau
        if ratio < -700.0:
            return 0.0
        decay = math.exp(ratio)
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"{name} decay must be in [0, 1)")
        return decay

    def _validated_gate(self, name: str) -> float:
        value = self._nonnegative(getattr(self, name), name)
        if value > self._GATE_MAX:
            raise ValueError(f"{name} outside Compte gate safety envelope")
        return value

    def _validated_state(self) -> tuple[float, float, float, float, float, float, float, float]:
        v = self._finite(self.v, "v")
        if not self._V_MIN <= v <= self._V_MAX:
            raise ValueError("v outside Compte WM safety envelope")
        s_ampa = self._validated_gate("s_ampa")
        s_nmda = self._validated_gate("s_nmda")
        if s_nmda > 1.0:
            raise ValueError("s_nmda must remain bounded by 1")
        x_nmda = self._validated_gate("x_nmda")
        s_gaba = self._validated_gate("s_gaba")

        self._nonnegative(self.g_l, "g_l")
        self._nonnegative(self.g_ampa, "g_ampa")
        self._nonnegative(self.g_nmda, "g_nmda")
        self._nonnegative(self.g_gaba, "g_gaba")
        self._finite(self.e_l, "e_l")
        self._finite(self.e_exc, "e_exc")
        self._finite(self.e_inh, "e_inh")
        self._positive(self.c_m, "c_m")
        self._nonnegative(self.mg, "mg")
        self._positive(self.tau_ampa, "tau_ampa")
        self._positive(self.tau_nmda, "tau_nmda")
        self._positive(self.tau_x, "tau_x")
        self._nonnegative(self.alpha_nmda, "alpha_nmda")
        self._finite(self.v_threshold, "v_threshold")
        self._finite(self.v_reset, "v_reset")
        if not self._V_MIN <= self.v_reset <= self._V_MAX:
            raise ValueError("v_reset outside Compte WM safety envelope")
        dt = self._positive(self.dt, "dt")
        return (
            v,
            s_ampa,
            s_nmda,
            x_nmda,
            s_gaba,
            self._decay(dt, self.tau_ampa, "AMPA"),
            self._decay(dt, self.tau_x, "NMDA x"),
            self._decay(dt, self._GABA_TAU, "GABA"),
        )

    def _mg_block(self, v: float) -> float:
        v = self._finite(v, "v")
        exponent = -0.062 * v
        exp_value = 0.0 if exponent < -700.0 else math.exp(min(exponent, 700.0))
        denominator = 1.0 + self.mg / 3.57 * exp_value
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ValueError("Mg block denominator must be finite and positive")
        block = 1.0 / denominator
        if not 0.0 <= block <= 1.0:
            raise ValueError("Mg block must be in [0, 1]")
        return block

    def step(self, current: float, spike_in: bool = False) -> int:
        current = self._finite(current, "current")
        v, s_ampa, s_nmda, x_nmda, s_gaba, decay_ampa, decay_x, decay_gaba = self._validated_state()

        s_ampa_pre = s_ampa + (1.0 if spike_in else 0.0)
        x_nmda_pre = x_nmda + (1.0 if spike_in else 0.0)
        if s_ampa_pre > self._GATE_MAX or x_nmda_pre > self._GATE_MAX:
            raise ValueError("spike input gate candidate outside Compte safety envelope")

        s_ampa_candidate = s_ampa_pre * decay_ampa
        s_nmda_candidate = (
            s_nmda
            + (-s_nmda / self.tau_nmda + self.alpha_nmda * x_nmda_pre * (1.0 - s_nmda)) * self.dt
        )
        x_nmda_candidate = x_nmda_pre * decay_x
        s_gaba_candidate = s_gaba * decay_gaba

        for value, name in (
            (s_ampa_candidate, "AMPA gate candidate"),
            (s_nmda_candidate, "NMDA gate candidate"),
            (x_nmda_candidate, "NMDA x candidate"),
            (s_gaba_candidate, "GABA gate candidate"),
        ):
            if not math.isfinite(value) or value < 0.0 or value > self._GATE_MAX:
                raise ValueError(f"{name} outside Compte safety envelope")
        if s_nmda_candidate > 1.0:
            raise ValueError("NMDA gate candidate must remain bounded by 1")

        b = self._mg_block(v)
        i_l = self.g_l * (v - self.e_l)
        i_ampa = self.g_ampa * s_ampa_candidate * (v - self.e_exc)
        i_nmda = self.g_nmda * b * s_nmda_candidate * (v - self.e_exc)
        i_gaba = self.g_gaba * s_gaba_candidate * (v - self.e_inh)
        dv = (-i_l - i_ampa - i_nmda - i_gaba + current) / self.c_m * self.dt
        v_candidate = v + dv

        for value, name in (
            (i_l, "leak current"),
            (i_ampa, "AMPA current"),
            (i_nmda, "NMDA current"),
            (i_gaba, "GABA current"),
            (dv, "voltage increment"),
            (v_candidate, "voltage candidate"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self._V_MIN <= v_candidate <= self._V_MAX:
            raise ValueError("voltage candidate outside Compte WM safety envelope")

        if v_candidate >= self.v_threshold:
            gaba_after_spike = s_gaba_candidate + 1.0
            if gaba_after_spike > self._GATE_MAX:
                raise ValueError("GABA spike candidate outside Compte safety envelope")
            self.v = self.v_reset
            self.s_ampa = s_ampa_candidate
            self.s_nmda = s_nmda_candidate
            self.x_nmda = x_nmda_candidate
            self.s_gaba = gaba_after_spike
            return 1

        self.v = v_candidate
        self.s_ampa = s_ampa_candidate
        self.s_nmda = s_nmda_candidate
        self.x_nmda = x_nmda_candidate
        self.s_gaba = s_gaba_candidate
        return 0

    def reset(self) -> None:
        self.v = self.e_l
        self.s_ampa = 0.0
        self.s_nmda = 0.0
        self.x_nmda = 0.0
        self.s_gaba = 0.0
