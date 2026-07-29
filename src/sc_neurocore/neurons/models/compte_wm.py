# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compte et al. 2000 pyramidal cell and synaptic kinetics

"""Source-bounded Compte working-memory pyramidal-cell dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy.typing as npt


@dataclass
class CompteWMNeuron:
    """Compte et al. pyramidal LIF cell with incoming AMPA/NMDA/GABAA gates.

    The class implements the excitatory-cell and channel equations from
    Compte et al., Cerebral Cortex 10(9), 910--923 (2000), DOI
    10.1093/cercor/10.9.910. external_spike increments the external AMPA gate,
    the compatibility argument spike_in increments the recurrent NMDA
    precursor, and inhibitory_spike increments the GABAA gate. A postsynaptic
    output spike resets the membrane and starts the source 2 ms refractory
    interval; it does not create an inhibitory autapse.

    Conductances use microSiemens, voltages millivolts, currents nanoamps,
    capacitance nanofarads, and time milliseconds. The default conductances
    are the paper's control-set pyramidal pathways: 3.1 nS external AMPA,
    0.381 nS recurrent NMDA, and 1.336 nS interneuron-to-pyramidal GABAA.
    One public step applies event jumps, then an explicit midpoint RK2 flow at
    the source 0.02 ms timestep. Threshold detection is sampled at the end of
    the step; the paper's within-step firing-time interpolation is not claimed.

    Notes
    -----
    This is not the paper's 2,560-cell ring network. Connectivity footprints,
    Poisson drive, tuned persistent bumps, and distractor statistics belong to
    a separately named SC project-derived network model.
    """

    v: float = -70.0
    s_ampa: float = 0.0
    s_nmda: float = 0.0
    x_nmda: float = 0.0
    s_gaba: float = 0.0
    g_l: float = 0.025
    g_ampa: float = 0.0031
    g_nmda: float = 0.000381
    g_gaba: float = 0.001336
    e_l: float = -70.0
    e_exc: float = 0.0
    e_inh: float = -70.0
    c_m: float = 0.5
    mg: float = 1.0
    tau_ampa: float = 2.0
    tau_nmda: float = 100.0
    tau_x: float = 2.0
    tau_gaba: float = 10.0
    alpha_nmda: float = 0.5
    v_threshold: float = -50.0
    v_reset: float = -60.0
    tau_ref: float = 2.0
    dt: float = 0.02

    _V_MIN = -200.0
    _V_MAX = 100.0
    _GATE_MAX = 1.0e6

    def __post_init__(self) -> None:
        """Normalise construction scalars and establish valid dynamic state."""
        for name in (
            "v",
            "s_ampa",
            "s_nmda",
            "x_nmda",
            "s_gaba",
            "g_l",
            "g_ampa",
            "g_nmda",
            "g_gaba",
            "e_l",
            "e_exc",
            "e_inh",
            "c_m",
            "mg",
            "tau_ampa",
            "tau_nmda",
            "tau_x",
            "tau_gaba",
            "alpha_nmda",
            "v_threshold",
            "v_reset",
            "tau_ref",
            "dt",
        ):
            setattr(self, name, float(getattr(self, name)))
        self._ref_remaining = 0.0
        self._validate()

    @staticmethod
    def _finite(name: str, value: float) -> float:
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return scalar

    @classmethod
    def _gate(cls, name: str, value: float) -> float:
        scalar = cls._finite(name, value)
        if scalar < 0.0 or scalar > cls._GATE_MAX:
            raise ValueError(f"{name} outside Compte gate safety envelope")
        return scalar

    def _validate(self) -> None:
        """Validate all mutable state and configuration without mutation."""
        v = self._finite("v", self.v)
        if not self._V_MIN <= v <= self._V_MAX:
            raise ValueError("v outside Compte WM safety envelope")
        self._gate("s_ampa", self.s_ampa)
        s_nmda = self._gate("s_nmda", self.s_nmda)
        self._gate("x_nmda", self.x_nmda)
        self._gate("s_gaba", self.s_gaba)
        if s_nmda > 1.0:
            raise ValueError("s_nmda must remain bounded by 1")
        for name in ("g_l", "g_ampa", "g_nmda", "g_gaba", "mg", "alpha_nmda"):
            if self._finite(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "c_m",
            "tau_ampa",
            "tau_nmda",
            "tau_x",
            "tau_gaba",
            "tau_ref",
            "dt",
        ):
            if self._finite(name, getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("e_l", "e_exc", "e_inh", "v_threshold", "v_reset"):
            self._finite(name, getattr(self, name))
        if not self._V_MIN <= self.v_reset <= self._V_MAX:
            raise ValueError("v_reset outside Compte WM safety envelope")
        if self._finite("_ref_remaining", self._ref_remaining) < 0.0:
            raise ValueError("_ref_remaining must be non-negative")

    def _mg_block(self, v: float) -> float:
        """Return the source Jahr--Stevens magnesium-unblock factor."""
        voltage = self._finite("v", v)
        exponent = -0.062 * voltage
        if exponent > 700.0:
            return 0.0
        block = 1.0 / (1.0 + self.mg / 3.57 * math.exp(exponent))
        if not math.isfinite(block) or not 0.0 <= block <= 1.0:
            raise FloatingPointError("Compte NMDA magnesium block is invalid")
        return block

    def _derivatives(
        self,
        v: float,
        s_ampa: float,
        s_nmda: float,
        x_nmda: float,
        s_gaba: float,
        current: float,
        *,
        membrane_active: bool,
    ) -> tuple[float, float, float, float, float]:
        """Return the coupled source derivatives for one RK2 stage."""
        d_ampa = -s_ampa / self.tau_ampa
        d_nmda = -s_nmda / self.tau_nmda + self.alpha_nmda * x_nmda * (1.0 - s_nmda)
        d_x = -x_nmda / self.tau_x
        d_gaba = -s_gaba / self.tau_gaba
        if membrane_active:
            i_l = self.g_l * (v - self.e_l)
            i_ampa = self.g_ampa * s_ampa * (v - self.e_exc)
            i_nmda = self.g_nmda * self._mg_block(v) * s_nmda * (v - self.e_exc)
            i_gaba = self.g_gaba * s_gaba * (v - self.e_inh)
            d_v = (-i_l - i_ampa - i_nmda - i_gaba + current) / self.c_m
        else:
            d_v = 0.0
        values = (d_v, d_ampa, d_nmda, d_x, d_gaba)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Compte RK2 derivative became non-finite")
        return values

    def step(
        self,
        current: float = 0.0,
        spike_in: bool = False,
        *,
        external_spike: bool = False,
        inhibitory_spike: bool = False,
    ) -> int:
        """Advance one atomic source-level midpoint-RK2 timestep.

        Parameters
        ----------
        current
            Direct somatic current in nA.
        spike_in
            Recurrent excitatory event. It increments the NMDA precursor and
            retains the historical positional argument name for compatibility.
        external_spike
            External excitatory event that increments the AMPA gate.
        inhibitory_spike
            Interneuron event that increments the GABAA gate.

        Returns
        -------
        int
            One for a sampled output spike, otherwise zero.

        Raises
        ------
        ValueError
            If mutable state, configuration, or current is invalid. Failure is
            atomic.
        FloatingPointError
            If an RK2 stage or candidate is non-finite. Failure is atomic.
        """
        current_value = self._finite("current", current)
        self._validate()
        v0 = self.v
        ref0 = self._ref_remaining
        s0 = (
            self.s_ampa + (1.0 if external_spike else 0.0),
            self.s_nmda,
            self.x_nmda + (1.0 if spike_in else 0.0),
            self.s_gaba + (1.0 if inhibitory_spike else 0.0),
        )
        for name, value in zip(("s_ampa", "s_nmda", "x_nmda", "s_gaba"), s0, strict=True):
            self._gate(f"{name} event candidate", value)

        membrane_active = ref0 <= 0.0
        k1 = self._derivatives(v0, *s0, current_value, membrane_active=membrane_active)
        midpoint = tuple(
            value + 0.5 * self.dt * slope for value, slope in zip((v0, *s0), k1, strict=True)
        )
        k2 = self._derivatives(
            midpoint[0],
            midpoint[1],
            midpoint[2],
            midpoint[3],
            midpoint[4],
            current_value,
            membrane_active=membrane_active,
        )
        candidate = tuple(
            value + self.dt * slope for value, slope in zip((v0, *s0), k2, strict=True)
        )
        v_next, ampa_next, nmda_next, x_next, gaba_next = candidate
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("Compte RK2 candidate became non-finite")
        if not self._V_MIN <= v_next <= self._V_MAX:
            raise ValueError("voltage candidate outside Compte WM safety envelope")
        for name, value in (
            ("AMPA gate candidate", ampa_next),
            ("NMDA gate candidate", nmda_next),
            ("NMDA precursor candidate", x_next),
            ("GABAA gate candidate", gaba_next),
        ):
            self._gate(name, value)
        if nmda_next > 1.0:
            raise ValueError("NMDA gate candidate must remain bounded by 1")

        event = 0
        ref_next = max(0.0, ref0 - self.dt)
        if not membrane_active:
            v_next = self.v_reset
        elif v_next >= self.v_threshold:
            v_next = self.v_reset
            ref_next = self.tau_ref
            event = 1

        self.v = v_next
        self.s_ampa = ampa_next
        self.s_nmda = nmda_next
        self.x_nmda = x_next
        self.s_gaba = gaba_next
        self._ref_remaining = ref_next
        return event

    def simulate(
        self,
        currents: npt.ArrayLike,
        recurrent_events: npt.ArrayLike,
        external_events: npt.ArrayLike,
        inhibitory_events: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> dict[str, object]:
        """Run complete state/event traces through one maintained backend."""
        from sc_neurocore.accel.compte_wm import simulate_compte_wm

        result = simulate_compte_wm(
            self.v,
            self.s_ampa,
            self.s_nmda,
            self.x_nmda,
            self.s_gaba,
            self._ref_remaining,
            self.g_l,
            self.g_ampa,
            self.g_nmda,
            self.g_gaba,
            self.e_l,
            self.e_exc,
            self.e_inh,
            self.c_m,
            self.mg,
            self.tau_ampa,
            self.tau_nmda,
            self.tau_x,
            self.tau_gaba,
            self.alpha_nmda,
            self.v_threshold,
            self.v_reset,
            self.tau_ref,
            self.dt,
            currents,
            recurrent_events,
            external_events,
            inhibitory_events,
            backend=backend,
        )
        self.v = float(result["v_final"])
        self.s_ampa = float(result["s_ampa_final"])
        self.s_nmda = float(result["s_nmda_final"])
        self.x_nmda = float(result["x_nmda_final"])
        self.s_gaba = float(result["s_gaba_final"])
        self._ref_remaining = float(result["ref_final"])
        return cast(dict[str, object], result)

    def reset(self) -> None:
        """Reset all dynamic state while preserving configuration."""
        self.v = self.e_l
        self.s_ampa = 0.0
        self.s_nmda = 0.0
        self.x_nmda = 0.0
        self.s_gaba = 0.0
        self._ref_remaining = 0.0

    def get_state(self) -> dict[str, float]:
        """Return the complete membrane, channel, and refractory state."""
        return {
            "v": self.v,
            "s_ampa": self.s_ampa,
            "s_nmda": self.s_nmda,
            "x_nmda": self.x_nmda,
            "s_gaba": self.s_gaba,
            "ref_remaining": self._ref_remaining,
        }


__all__ = ["CompteWMNeuron"]
