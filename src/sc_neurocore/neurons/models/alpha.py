# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dual alpha-synapse leaky integrate-and-fire neuron

"""Exact constant-input flow for the dual alpha-synapse LIF."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

AlphaResult = dict[str, npt.NDArray[np.float64] | float | int]


@dataclass
class AlphaNeuron:
    """Dual excitatory/inhibitory alpha-synapse leaky integrate-and-fire neuron.

    The membrane equation is the leaky integrate-and-fire relaxation

    ``tau_v * dV/dt = -(V - v_rest) + i_exc - i_inh``,

    and each synaptic current is carried by a two-state alpha cascade
    (rise ``a``, current ``i``) that reproduces Rall's alpha kernel
    ``alpha(t) ~ (t/tau) * exp(1 - t/tau)`` for a pulse input. A spike is
    emitted when the candidate membrane potential reaches ``v_threshold``;
    only the membrane potential resets (``v <- v_rest``); the synaptic
    cascade states are preserved across spikes.

    The maintained numerical step is the exact piecewise-constant-input
    flow: each alpha filter relaxes exactly, and the membrane update
    integrates the alpha currents with the exact convolution, including the
    equal-time-constant limit. This exact flow is the engineering contract,
    not a biological publication claim.

    Defaults ``tau_v=20``, ``tau_exc=5``, ``tau_inh=10``, ``v_rest=0``,
    ``v_threshold=1``, and ``dt=1.0`` are catalogue/model-family choices,
    not source-derived parameters.

    References
    ----------
    Rall, W. (1967). Distinguishing theoretical synaptic potentials computed
    for different soma-dendritic distributions of synaptic input. Journal of
    Neurophysiology 30(5), 1138–1168. (The alpha kernel.)

    Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models. Cambridge
    University Press, §4.1. https://doi.org/10.1017/CBO9780511815706
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
        """Normalise scalar fields and reject an invalid configuration."""
        for name in (
            "v",
            "i_exc",
            "i_inh",
            "a_exc",
            "a_inh",
            "v_rest",
            "v_threshold",
            "tau_v",
            "tau_exc",
            "tau_inh",
            "dt",
        ):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            setattr(self, name, value)
        self._validated_state()
        self._validated_parameters()

    def _validated_state(self) -> tuple[float, float, float, float, float]:
        """Return the current finite membrane and synaptic cascade states."""
        try:
            values = (
                float(self.v),
                float(self.a_exc),
                float(self.i_exc),
                float(self.a_inh),
                float(self.i_inh),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("alpha state must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("alpha state must be finite")
        return values

    def _validated_parameters(self) -> tuple[float, float, float, float, float, float]:
        """Return the finite numerical configuration without mutation."""
        try:
            values = (
                float(self.v_rest),
                float(self.v_threshold),
                float(self.tau_v),
                float(self.tau_exc),
                float(self.tau_inh),
                float(self.dt),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("alpha parameters must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("alpha parameters must be finite")
        v_rest, v_threshold, tau_v, tau_exc, tau_inh, dt = values
        if tau_v <= 0.0:
            raise ValueError("tau_v must be positive")
        if tau_exc <= 0.0:
            raise ValueError("tau_exc must be positive")
        if tau_inh <= 0.0:
            raise ValueError("tau_inh must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if v_threshold <= v_rest:
            raise ValueError("v_threshold must be greater than v_rest")
        return values

    @staticmethod
    def _filter_candidates(
        rise_state: float, current_state: float, drive: float, tau: float, dt: float
    ) -> tuple[float, float]:
        """Return the exact constant-input alpha-filter candidate pair."""
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
        """Return the exact alpha-current convolution over one ``dt`` interval."""
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

    def _candidate(
        self, exc_current: float, inh_current: float
    ) -> tuple[float, float, float, float, float, int]:
        """Compute one validated candidate without mutating caller-visible state."""
        try:
            exc_drive = float(exc_current)
            inh_drive = float(inh_current)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current values must be numeric") from exc
        if not math.isfinite(exc_drive) or not math.isfinite(inh_drive):
            raise ValueError("current values must be finite")
        v, a_exc, i_exc, a_inh, i_inh = self._validated_state()
        (
            v_rest,
            _v_threshold,
            tau_v,
            tau_exc,
            tau_inh,
            dt,
        ) = self._validated_parameters()

        exc_steady = tau_exc * exc_drive
        inh_steady = tau_inh * inh_drive
        exc_rise_delta = a_exc - exc_steady
        inh_rise_delta = a_inh - inh_steady
        exc_current_delta = i_exc - exc_steady
        inh_current_delta = i_inh - inh_steady

        a_exc_next, i_exc_next = self._filter_candidates(a_exc, i_exc, exc_drive, tau_exc, dt)
        a_inh_next, i_inh_next = self._filter_candidates(a_inh, i_inh, inh_drive, tau_inh, dt)
        v_steady = v_rest + exc_steady - inh_steady
        decay_v = math.exp(-dt / tau_v)
        v_next = (
            v_steady
            + (v - v_steady) * decay_v
            + self._drive_contribution(exc_current_delta, exc_rise_delta, tau_exc, tau_v, dt)
            - self._drive_contribution(inh_current_delta, inh_rise_delta, tau_inh, tau_v, dt)
        )
        if not all(
            math.isfinite(value)
            for value in (a_exc_next, i_exc_next, a_inh_next, i_inh_next, v_next)
        ):
            raise FloatingPointError("alpha exact-flow candidate must be finite")
        if v_next >= _v_threshold:
            return a_exc_next, i_exc_next, a_inh_next, i_inh_next, v_rest, 1
        return a_exc_next, i_exc_next, a_inh_next, i_inh_next, v_next, 0

    def step(self, exc_current: float, inh_current: float = 0.0) -> int:
        """Advance one exact-flow interval and return a binary spike event.

        Mutation is atomic: invalid input, configuration, or candidate state
        leaves every dynamic state unchanged.
        """
        (
            a_exc_next,
            i_exc_next,
            a_inh_next,
            i_inh_next,
            v_next,
            spike,
        ) = self._candidate(exc_current, inh_current)
        self.a_exc = a_exc_next
        self.i_exc = i_exc_next
        self.a_inh = a_inh_next
        self.i_inh = i_inh_next
        self.v = v_next
        return spike

    def simulate(
        self,
        exc_current: npt.ArrayLike,
        inh_current: npt.ArrayLike | float = 0.0,
        *,
        backend: str = "auto",
    ) -> AlphaResult:
        """Run one atomic piecewise-constant-input batch on a maintained backend."""
        from sc_neurocore.accel.alpha import simulate_alpha

        result = simulate_alpha(
            self.v,
            self.a_exc,
            self.i_exc,
            self.a_inh,
            self.i_inh,
            self.v_rest,
            self.v_threshold,
            self.tau_v,
            self.tau_exc,
            self.tau_inh,
            self.dt,
            exc_current,
            inh_current,
            backend=backend,
        )
        self.v = float(cast(float, result["v_final"]))
        self.a_exc = float(cast(float, result["a_exc_final"]))
        self.i_exc = float(cast(float, result["i_exc_final"]))
        self.a_inh = float(cast(float, result["a_inh_final"]))
        self.i_inh = float(cast(float, result["i_inh_final"]))
        return result

    def reset(self) -> None:
        """Restore the documented rest state while preserving configuration."""
        self.v = self.v_rest
        self.a_exc = 0.0
        self.i_exc = 0.0
        self.a_inh = 0.0
        self.i_inh = 0.0


__all__ = ["AlphaNeuron", "AlphaResult"]
