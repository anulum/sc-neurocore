# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel-Wang LIF with NMDA/AMPA/GABA synapses

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class BrunelWangNeuron:
    """LIF neuron with NMDA, AMPA, and GABA synaptic currents.

    Reference: Brunel, N. & Wang, X.J. (2001). Effects of neuromodulation in a
    cortical network model of object working memory dominated by
    recurrent inhibition. J Comput Neurosci 11:63-85.

    Used in decision-making and working memory models. The key feature
    is the voltage-dependent NMDA conductance with Mg2+ block.
    """

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -55.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    tau_ref: float = 2.0
    tau_ampa: float = 2.0
    tau_nmda_rise: float = 2.0
    tau_nmda_decay: float = 100.0
    tau_gaba: float = 5.0
    g_ampa_ext: float = 2.1
    g_ampa_rec: float = 0.05
    g_nmda: float = 0.165
    g_gaba: float = 1.3
    v_ampa: float = 0.0
    v_nmda: float = 0.0
    v_gaba: float = -70.0
    C_m: float = 0.5
    mg_conc: float = 1.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "v",
            "v_rest",
            "v_reset",
            "v_threshold",
            "tau_m",
            "tau_ref",
            "tau_ampa",
            "tau_nmda_rise",
            "tau_nmda_decay",
            "tau_gaba",
            "g_ampa_ext",
            "g_ampa_rec",
            "g_nmda",
            "g_gaba",
            "v_ampa",
            "v_nmda",
            "v_gaba",
            "C_m",
            "mg_conc",
            "dt",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in (
            "tau_m",
            "tau_ref",
            "tau_ampa",
            "tau_nmda_rise",
            "tau_nmda_decay",
            "tau_gaba",
            "C_m",
            "dt",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("g_ampa_ext", "g_ampa_rec", "g_nmda", "g_gaba", "mg_conc"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self._validate_voltage(self.v)
        self._s_ampa = 0.0
        self._s_nmda = 0.0
        self._x_nmda = 0.0
        self._s_gaba = 0.0
        self._ref_remaining = 0.0

    @staticmethod
    def _validate_voltage(v: float) -> float:
        voltage = float(v)
        if not math.isfinite(voltage):
            raise FloatingPointError("Brunel-Wang voltage state must be finite")
        return voltage

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> float:
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar

    @staticmethod
    def _validate_gate(name: str, value: float) -> float:
        gate = float(value)
        if not math.isfinite(gate) or gate < 0.0 or gate > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return gate

    def _nmda_voltage_dep(self, v: float) -> float:
        """Mg2+ block factor: 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))."""
        voltage = self._validate_voltage(v)
        exponent = -0.062 * voltage
        if exponent > 700.0:
            return 0.0
        factor = 1.0 / (1.0 + self.mg_conc / 3.57 * math.exp(exponent))
        if not math.isfinite(factor) or factor < 0.0 or factor > 1.0:
            raise FloatingPointError("Brunel-Wang NMDA voltage factor is invalid")
        return factor

    def step(
        self,
        i_ampa_ext: float = 0.0,
        s_ampa_rec: float = 0.0,
        s_nmda_rec: float = 0.0,
        s_gaba: float = 0.0,
    ) -> int:
        """Advance one timestep.

        Parameters
        ----------
        i_ampa_ext : external AMPA current (from Poisson input)
        s_ampa_rec : recurrent AMPA synaptic variable [0, 1]
        s_nmda_rec : recurrent NMDA synaptic variable [0, 1]
        s_gaba : inhibitory GABA synaptic variable [0, 1]
        """
        ampa_ext = self._validate_nonnegative("i_ampa_ext", i_ampa_ext)
        ampa_rec = self._validate_gate("s_ampa_rec", s_ampa_rec)
        nmda_rec = self._validate_gate("s_nmda_rec", s_nmda_rec)
        gaba = self._validate_gate("s_gaba", s_gaba)
        v = self._validate_voltage(self.v)
        ref_remaining = self._validate_nonnegative("_ref_remaining", self._ref_remaining)

        if ref_remaining > 0:
            self._ref_remaining = max(0.0, ref_remaining - self.dt)
            return 0

        # Synaptic currents
        i_ampa = -self.g_ampa_ext * (v - self.v_ampa) * ampa_ext
        i_ampa += -self.g_ampa_rec * (v - self.v_ampa) * ampa_rec
        i_nmda = -self.g_nmda * self._nmda_voltage_dep(v) * (v - self.v_nmda) * nmda_rec
        i_gaba = -self.g_gaba * (v - self.v_gaba) * gaba

        # Membrane dynamics
        i_leak = -(v - self.v_rest) / self.tau_m
        dv = (i_leak + (i_ampa + i_nmda + i_gaba) / self.C_m) * self.dt
        next_v = v + dv
        if not all(math.isfinite(term) for term in (i_ampa, i_nmda, i_gaba, i_leak, dv, next_v)):
            raise FloatingPointError("Brunel-Wang membrane candidate became non-finite")

        self.v = next_v

        if next_v >= self.v_threshold:
            self.v = self.v_reset
            self._ref_remaining = self.tau_ref
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self._s_ampa = 0.0
        self._s_nmda = 0.0
        self._x_nmda = 0.0
        self._s_gaba = 0.0
        self._ref_remaining = 0.0

    def get_state(self) -> dict[str, float]:
        return {"v": self.v, "ref_remaining": self._ref_remaining}
