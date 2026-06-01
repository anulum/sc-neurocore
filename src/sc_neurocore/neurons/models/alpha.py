# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha-synapse neuron. Rall 1967

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class AlphaNeuron:
    """Alpha-synapse neuron. Rall 1967.

    Dual excitatory/inhibitory synaptic currents with alpha-function kinetics.

    Reference: Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models. Cambridge Univ. Press, §4.1.
    """

    v: float = 0.0
    i_exc: float = 0.0
    i_inh: float = 0.0
    a_exc: float = 0.0
    a_inh: float = 0.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    tau_v: float = 20.0
    tau_exc: float = 5.0
    tau_inh: float = 10.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._validate_state()

    def _validate_state(self) -> None:
        for field in ("v", "a_exc", "i_exc", "a_inh", "i_inh", "v_rest", "v_threshold"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("tau_v", "tau_exc", "tau_inh", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    @staticmethod
    def _filter_candidates(
        rise_state: float, current_state: float, drive: float, tau: float, dt: float
    ) -> tuple[float, float]:
        steady_state = tau * drive
        rise_delta = rise_state - steady_state
        current_delta = current_state - steady_state
        decay = math.exp(-dt / tau)
        rise_next = steady_state + rise_delta * decay
        current_next = steady_state + decay * (current_delta + rise_delta * dt / tau)
        return rise_next, current_next

    @staticmethod
    def _drive_contribution(
        current_delta: float, rise_delta: float, tau_drive: float, tau_v: float, dt: float
    ) -> float:
        rate_v = 1.0 / tau_v
        rate_drive = 1.0 / tau_drive
        decay_v = math.exp(-dt / tau_v)
        decay_drive = math.exp(-dt / tau_drive)
        if math.isclose(rate_v, rate_drive, rel_tol=0.0, abs_tol=1.0e-14):
            return (
                rate_v * decay_v * (current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive))
            )
        rate_delta = rate_v - rate_drive
        first_order = current_delta * (decay_drive - decay_v) / rate_delta
        second_order = (
            rise_delta
            / tau_drive
            * (decay_drive * (rate_delta * dt - 1.0) + decay_v)
            / (rate_delta * rate_delta)
        )
        return rate_v * (first_order + second_order)

    def step(self, exc_current: float, inh_current: float = 0.0) -> int:
        if not math.isfinite(exc_current) or not math.isfinite(inh_current):
            raise ValueError("current values must be finite")
        self._validate_state()

        exc_steady = self.tau_exc * exc_current
        inh_steady = self.tau_inh * inh_current
        exc_rise_delta = self.a_exc - exc_steady
        inh_rise_delta = self.a_inh - inh_steady
        exc_current_delta = self.i_exc - exc_steady
        inh_current_delta = self.i_inh - inh_steady

        a_exc_next, i_exc_next = self._filter_candidates(
            self.a_exc, self.i_exc, exc_current, self.tau_exc, self.dt
        )
        a_inh_next, i_inh_next = self._filter_candidates(
            self.a_inh, self.i_inh, inh_current, self.tau_inh, self.dt
        )
        v_steady = self.v_rest + exc_steady - inh_steady
        v_next = (
            v_steady
            + (self.v - v_steady) * math.exp(-self.dt / self.tau_v)
            + self._drive_contribution(
                exc_current_delta, exc_rise_delta, self.tau_exc, self.tau_v, self.dt
            )
            - self._drive_contribution(
                inh_current_delta, inh_rise_delta, self.tau_inh, self.tau_v, self.dt
            )
        )
        if not all(
            math.isfinite(value)
            for value in (a_exc_next, i_exc_next, a_inh_next, i_inh_next, v_next)
        ):
            raise ValueError("alpha exact-flow update became non-finite")

        self.a_exc = a_exc_next
        self.i_exc = i_exc_next
        self.a_inh = a_inh_next
        self.i_inh = i_inh_next
        if v_next >= self.v_threshold:
            self.v = self.v_rest
            return 1
        self.v = v_next
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.a_exc = 0.0
        self.i_exc = 0.0
        self.a_inh = 0.0
        self.i_inh = 0.0
