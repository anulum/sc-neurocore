# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Loihi CUBA LIF — Intel Loihi fixed-point neuron. Davies 2018

"""Intel Loihi CUBA leaky integrate-and-fire neuron, in integer arithmetic."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from sc_neurocore.utils.numerics import trunc_div

_FIELD_NAMES = ("v", "u", "tau_v", "tau_u", "v_threshold", "v_reset")


def _validate_finite_real(value: object, name: str) -> None:
    """Reject booleans, non-real types and non-finite real values."""
    if type(value) is int:
        return
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number, not bool")
    if isinstance(value, numbers.Integral):
        return
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return
    raise TypeError(f"{name} must be a finite real number")


@dataclass
class LoihiCUBANeuron:
    """Loihi CUBA LIF — Intel Loihi fixed-point neuron. Davies 2018.

    Each step ``u`` decays by ``u / tau_u`` and gains the input, then ``v`` decays by ``v / tau_v`` and gains ``u``.
    The quotients are truncated towards zero, as the engine and Loihi's sign-magnitude decay round them, so a negative
    state decays as its positive mirror does.

    Reference: Davies, M. et al. (2018). IEEE Micro 38:82–99.
    """

    v: int = 0
    u: int = 0
    tau_v: int = 10
    tau_u: int = 5
    v_threshold: int = 1000
    v_reset: int = 0

    def __post_init__(self) -> None:
        """Validate all state and parameter fields and require decay divisors >= 1."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name in _FIELD_NAMES:
            value: object = getattr(self, name)
            _validate_finite_real(value, name)
            if type(value) is not int and isinstance(value, numbers.Integral):
                setattr(self, name, int(value))
        for name in ("tau_v", "tau_u"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")

    def _step_validated(self, weighted_input: int) -> int:
        """Validate non-integer inputs or mutated fields and advance transactionally."""
        _validate_finite_real(weighted_input, "weighted_input")
        if isinstance(weighted_input, numbers.Integral):
            weighted_input = int(weighted_input)
        self._validate_configuration()

        try:
            next_u = self.u - trunc_div(self.u, self.tau_u) + weighted_input
            next_v = self.v - trunc_div(self.v, self.tau_v) + next_u
        except OverflowError as exc:
            raise FloatingPointError(
                "Loihi CUBA state overflowed finite floating-point range"
            ) from exc

        if (next_u.__class__ is not int and not math.isfinite(next_u)) or (
            next_v.__class__ is not int and not math.isfinite(next_v)
        ):
            raise FloatingPointError("Loihi CUBA state overflowed finite floating-point range")

        self.u = next_u
        if next_v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        self.v = next_v
        return 0

    def step(self, weighted_input: int) -> int:
        """Advance one chip step with the integer input; return 1 on a spike, else 0.

        Parameters
        ----------
        weighted_input : int
            Synaptic drive added to ``u``. Exact integer inputs preserve
            integer state; finite real floats are accepted for network drivers.

        Returns
        -------
        int
            1 when ``v`` reaches or exceeds ``v_threshold``, else 0.

        Raises
        ------
        ValueError
            If ``weighted_input`` or any state/parameter is non-finite, or
            ``tau_v`` or ``tau_u`` is less than 1. State is left unchanged.
        TypeError
            If ``weighted_input`` or any state/parameter is a boolean or
            non-real value. State is left unchanged.
        FloatingPointError
            If a floating-point candidate state overflows finite range.
            State is left unchanged.
        """
        v = self.v
        u = self.u
        tau_v = self.tau_v
        tau_u = self.tau_u
        v_threshold = self.v_threshold
        v_reset = self.v_reset
        if (
            weighted_input.__class__ is int
            and v.__class__ is int
            and u.__class__ is int
            and tau_v.__class__ is int
            and tau_u.__class__ is int
            and v_threshold.__class__ is int
            and v_reset.__class__ is int
            and tau_v >= 1
            and tau_u >= 1
        ):
            next_u = u - trunc_div(u, tau_u) + weighted_input
            next_v = v - trunc_div(v, tau_v) + next_u
            self.u = next_u
            if next_v >= v_threshold:
                self.v = v_reset
                return 1
            self.v = next_v
            return 0
        return self._step_validated(weighted_input)

    def reset(self) -> None:
        """Clear both states; the parameters stay."""
        self.v, self.u = 0, 0
