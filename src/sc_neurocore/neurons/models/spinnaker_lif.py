# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SpiNNaker LIF — ARM Cortex-M4 digital. Furber 2014

from __future__ import annotations

from dataclasses import dataclass
import math


def _finite_scalar(name: str, value: float) -> float:
    """Return ``value`` as a finite float or raise a typed validation error."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


@dataclass
class SpiNNakerLIFNeuron:
    """SpiNNaker LIF with exact constant-current membrane flow.

    The SpiNNaker software LIF model evolves one membrane state ``v`` under a
    constant input current plus tonic ``i_offset`` during each integration
    interval. SC-NeuroCore evaluates that linear ODE analytically instead of
    using forward Euler, while retaining the documented hard threshold, reset,
    and absolute refractory timer semantics.

    References
    ----------
    Furber, S. B. et al. (2014). The SpiNNaker Project. Proceedings of the
    IEEE, 102(5), 652-665.
    """

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    i_offset: float = 0.0
    tau_refrac: float = 2.0
    refrac_count: float = 0.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalise public scalar parameters."""

        self.v = _finite_scalar("v", self.v)
        self.v_rest = _finite_scalar("v_rest", self.v_rest)
        self.v_reset = _finite_scalar("v_reset", self.v_reset)
        self.v_threshold = _finite_scalar("v_threshold", self.v_threshold)
        self.tau_m = _finite_scalar("tau_m", self.tau_m)
        self.i_offset = _finite_scalar("i_offset", self.i_offset)
        self.tau_refrac = _finite_scalar("tau_refrac", self.tau_refrac)
        self.refrac_count = _finite_scalar("refrac_count", self.refrac_count)
        self.dt = _finite_scalar("dt", self.dt)
        if self.tau_m <= 0.0:
            raise ValueError("tau_m must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.tau_refrac < 0.0:
            raise ValueError("tau_refrac must be non-negative")
        if self.refrac_count < 0.0:
            raise ValueError("refrac_count must be non-negative")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must exceed v_reset")

    def step(self, current: float) -> int:
        """Advance one exact-flow step and return a binary spike indicator."""

        current = _finite_scalar("current", current)
        self._validate_runtime_state()
        if self.refrac_count > 0:
            self.refrac_count = max(0.0, self.refrac_count - self.dt)
            return 0

        next_v = self._exact_membrane_candidate(current)
        if not math.isfinite(next_v):
            raise ValueError("exact membrane candidate must be finite")
        if next_v >= self.v_threshold:
            self.v = self.v_reset
            self.refrac_count = self.tau_refrac
            return 1
        self.v = next_v
        return 0

    def reset(self) -> None:
        """Restore voltage and refractory state to the documented rest state."""

        self.v = self.v_rest
        self.refrac_count = 0.0

    def _exact_membrane_candidate(self, current: float) -> float:
        """Return the exact membrane solution for one constant-current step."""

        steady = self.v_rest + current + self.i_offset
        decay = math.exp(-self.dt / self.tau_m)
        return steady + (self.v - steady) * decay

    def _validate_runtime_state(self) -> None:
        """Reject corrupted runtime state before mutating the neuron."""

        for name in ("v", "refrac_count"):
            value = _finite_scalar(name, getattr(self, name))
            if name == "refrac_count" and value < 0.0:
                raise ValueError("refrac_count must be non-negative")


# ── SPECIALIZED / MODERN ──────────────────────────────────────────
