# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Merolla 2014 — IBM TrueNorth digital neuron

"""IBM TrueNorth digital integrate-and-fire neuron in integer arithmetic."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

_FIELD_NAMES = ("v", "leak", "threshold", "v_reset")


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
class TrueNorthNeuron:
    """Merolla 2014 — IBM TrueNorth digital neuron.

    Reference: Merolla, P.A. et al. (2014). Science 345:668–673.
    """

    v: int = 0
    leak: int = 0
    threshold: int = 100
    v_reset: int = 0

    def __post_init__(self) -> None:
        """Validate all state and parameter fields and require a positive threshold."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name in _FIELD_NAMES:
            value: object = getattr(self, name)
            _validate_finite_real(value, name)
            if type(value) is not int and isinstance(value, numbers.Integral):
                setattr(self, name, int(value))
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")

    def _step_validated(self, weighted_input: int) -> int:
        """Validate non-integer inputs or mutated fields and advance transactionally."""
        _validate_finite_real(weighted_input, "weighted_input")
        if isinstance(weighted_input, numbers.Integral):
            weighted_input = int(weighted_input)
        self._validate_configuration()

        try:
            next_v = self.v + weighted_input - self.leak
        except OverflowError as exc:
            raise FloatingPointError(
                "TrueNorth membrane potential overflowed finite floating-point range"
            ) from exc

        if next_v.__class__ is not int and not math.isfinite(next_v):
            raise FloatingPointError(
                "TrueNorth membrane potential overflowed finite floating-point range"
            )

        if next_v >= self.threshold:
            self.v = self.v_reset
            return 1
        if next_v < -self.threshold:
            self.v = self.v_reset
            return 0
        self.v = next_v
        return 0

    def step(self, weighted_input: int) -> int:
        """Advance one digital step with the weighted input; return 1 on a spike, else 0.

        Parameters
        ----------
        weighted_input : int
            Synaptic input added to ``v`` before subtracting ``leak``. Exact
            integer inputs preserve integer state; finite real floats are
            accepted for network drivers.

        Returns
        -------
        int
            1 when ``v`` reaches or exceeds ``threshold``, else 0.

        Raises
        ------
        ValueError
            If ``weighted_input`` or any state/parameter is non-finite, or
            ``threshold`` is not positive. State is left unchanged.
        TypeError
            If ``weighted_input`` or any state/parameter is a boolean or
            non-real value. State is left unchanged.
        FloatingPointError
            If a floating-point candidate membrane potential overflows finite
            range. State is left unchanged.
        """
        v = self.v
        leak = self.leak
        threshold = self.threshold
        v_reset = self.v_reset
        if (
            weighted_input.__class__ is int
            and v.__class__ is int
            and leak.__class__ is int
            and threshold.__class__ is int
            and v_reset.__class__ is int
            and threshold > 0
        ):
            next_v = v + weighted_input - leak
            if next_v >= threshold:
                self.v = v_reset
                return 1
            if next_v < -threshold:
                self.v = v_reset
                return 0
            self.v = next_v
            return 0
        return self._step_validated(weighted_input)

    def reset(self) -> None:
        """Reset membrane potential to 0; the parameters stay."""
        self.v = 0
