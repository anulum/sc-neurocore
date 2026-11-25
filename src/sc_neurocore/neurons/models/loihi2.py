# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Intel Loihi 2, 2021 — programmable 3-state-variable neuron

"""Intel Loihi 2 three-state integer neuron with cross-coupling and a decay truncated toward zero."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from sc_neurocore.utils.numerics import trunc_div

_FIELD_NAMES = (
    "s1",
    "s2",
    "s3",
    "tau1",
    "tau2",
    "tau3",
    "w12",
    "w13",
    "w23",
    "s1_threshold",
    "s1_reset",
    "s3_incr",
)


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
class Loihi2Neuron:
    """Intel Loihi 2, 2021 — programmable 3-state-variable neuron.

    State variables (s1, s2, s3) with configurable decay, threshold,
    and cross-coupling. Generalises CUBA, COBA, and Izhikevich on-chip.
    All integer arithmetic: each state decays by its quotient by its time
    constant, truncated towards zero as the engine and Loihi's
    sign-magnitude decay round it, so a negative state decays as its
    positive mirror does.

    Reference: Intel Corp. (2021). Loihi 2 Neuromorphic Processor Technical Brief.
    """

    s1: int = 0  # primary state (membrane potential analogue)
    s2: int = 0  # secondary state (synaptic current / adaptation)
    s3: int = 0  # tertiary state (slow modulation)
    tau1: int = 10  # s1 decay divisor
    tau2: int = 5  # s2 decay divisor
    tau3: int = 50  # s3 decay divisor
    w12: int = 1  # s2 → s1 coupling weight
    w13: int = 0  # s3 → s1 coupling weight
    w23: int = 0  # s3 → s2 coupling weight
    s1_threshold: int = 1000
    s1_reset: int = 0
    s3_incr: int = 10  # s3 increment on spike (adaptation)

    def __post_init__(self) -> None:
        """Validate all state and parameter fields and require decay divisors >= 1."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name in _FIELD_NAMES:
            value: object = getattr(self, name)
            _validate_finite_real(value, name)
            if type(value) is not int and isinstance(value, numbers.Integral):
                setattr(self, name, int(value))
        for name in ("tau1", "tau2", "tau3"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")

    def _step_validated(self, weighted_input: int) -> int:
        """Validate non-integer inputs or mutated fields and advance transactionally."""
        _validate_finite_real(weighted_input, "weighted_input")
        if isinstance(weighted_input, numbers.Integral):
            weighted_input = int(weighted_input)
        self._validate_configuration()

        try:
            next_s3 = self.s3 - trunc_div(self.s3, self.tau3)
            next_s2 = self.s2 - trunc_div(self.s2, self.tau2) + weighted_input + self.w23 * next_s3
            next_s1 = (
                self.s1 - trunc_div(self.s1, self.tau1) + self.w12 * next_s2 + self.w13 * next_s3
            )
            spiked = next_s1 >= self.s1_threshold
            if spiked:
                next_s3 = next_s3 + self.s3_incr
        except OverflowError as exc:
            raise FloatingPointError(
                "Loihi 2 state overflowed finite floating-point range"
            ) from exc

        if (
            (next_s1.__class__ is not int and not math.isfinite(next_s1))
            or (next_s2.__class__ is not int and not math.isfinite(next_s2))
            or (next_s3.__class__ is not int and not math.isfinite(next_s3))
        ):
            raise FloatingPointError("Loihi 2 state overflowed finite floating-point range")

        self.s2 = next_s2
        self.s3 = next_s3
        if spiked:
            self.s1 = self.s1_reset
            return 1
        self.s1 = next_s1
        return 0

    def step(self, weighted_input: int) -> int:
        """Advance one chip step with the integer input; return 1 on a spike, else 0.

        Parameters
        ----------
        weighted_input : int
            Synaptic drive added to ``s2``. Exact integer inputs preserve
            integer state; finite real floats are accepted for network drivers.

        Returns
        -------
        int
            1 when ``s1`` reaches or exceeds ``s1_threshold``, else 0.

        Raises
        ------
        ValueError
            If ``weighted_input`` or any state/parameter is non-finite, or any
            decay divisor (``tau1``, ``tau2``, ``tau3``) is less than 1.
            State is left unchanged.
        TypeError
            If ``weighted_input`` or any state/parameter is a boolean or
            non-real value. State is left unchanged.
        FloatingPointError
            If a floating-point candidate state overflows finite range.
            State is left unchanged.
        """
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        tau1 = self.tau1
        tau2 = self.tau2
        tau3 = self.tau3
        w12 = self.w12
        w13 = self.w13
        w23 = self.w23
        s1_threshold = self.s1_threshold
        s1_reset = self.s1_reset
        s3_incr = self.s3_incr
        if (
            weighted_input.__class__ is int
            and s1.__class__ is int
            and s2.__class__ is int
            and s3.__class__ is int
            and tau1.__class__ is int
            and tau2.__class__ is int
            and tau3.__class__ is int
            and w12.__class__ is int
            and w13.__class__ is int
            and w23.__class__ is int
            and s1_threshold.__class__ is int
            and s1_reset.__class__ is int
            and s3_incr.__class__ is int
            and tau1 >= 1
            and tau2 >= 1
            and tau3 >= 1
        ):
            next_s3 = s3 - trunc_div(s3, tau3)
            next_s2 = s2 - trunc_div(s2, tau2) + weighted_input + w23 * next_s3
            next_s1 = s1 - trunc_div(s1, tau1) + w12 * next_s2 + w13 * next_s3
            self.s2 = next_s2
            if next_s1 >= s1_threshold:
                self.s1 = s1_reset
                self.s3 = next_s3 + s3_incr
                return 1
            self.s1 = next_s1
            self.s3 = next_s3
            return 0
        return self._step_validated(weighted_input)

    def reset(self) -> None:
        """Clear the three states; the parameters stay."""
        self.s1, self.s2, self.s3 = 0, 0, 0
