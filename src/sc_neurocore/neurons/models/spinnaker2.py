# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — TU Dresden / SpiNNaker2 2024 — ARM Cortex-M4F software LIF

"""TU Dresden / SpiNNaker2 fixed-point integer leaky integrate-and-fire neuron."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass


def _require_int(value: object, name: str) -> int:
    """Validate that ``value`` is a non-boolean integer and return it as ``int``."""
    if type(value) is int:
        return value
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    raise TypeError(f"{name} must be an integer")


@dataclass
class SpiNNaker2Neuron:
    """TU Dresden / SpiNNaker2 2024 — ARM Cortex-M4F software LIF.

    Fixed-point LIF on M4F with exponential decay via integer multiply-shift.
    Includes refractory counter and configurable precision.

    Reference: Mayr, C. et al. (2019). Proc. DATE 2019: 1547–1552.
    """

    v: int = 0
    v_rest: int = 0
    v_reset: int = 0
    v_threshold: int = 1024
    decay_mult: int = 243  # ~exp(-1/10) * 256, 8-bit fixed-point
    decay_shift: int = 8  # right-shift after multiply
    refrac_steps: int = 2
    _refrac_count: int = 0

    def __post_init__(self) -> None:
        """Validate all fixed-point fields and coerce integral values to ``int``."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.v = _require_int(self.v, "v")
        self.v_rest = _require_int(self.v_rest, "v_rest")
        self.v_reset = _require_int(self.v_reset, "v_reset")
        self.v_threshold = _require_int(self.v_threshold, "v_threshold")
        self.decay_mult = _require_int(self.decay_mult, "decay_mult")
        self.decay_shift = _require_int(self.decay_shift, "decay_shift")
        self.refrac_steps = _require_int(self.refrac_steps, "refrac_steps")
        self._refrac_count = _require_int(self._refrac_count, "_refrac_count")
        for name in ("decay_mult", "decay_shift", "refrac_steps", "_refrac_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def step(self, current: int) -> int:
        """Advance one fixed-point step with an integer input; return 1 on a spike, else 0.

        Parameters
        ----------
        current : int
            Integer synaptic drive added after the fixed-point multiply-shift decay.

        Returns
        -------
        int
            1 when the membrane reaches or exceeds ``v_threshold``, else 0.

        Raises
        ------
        ValueError
            If ``current`` or any neuron parameter/state is non-finite, or a
            non-negative configuration field is negative. State is left unchanged.
        TypeError
            If ``current`` or any neuron parameter/state is a boolean, float or
            other non-integer value. State is left unchanged.
        """
        v = self.v
        v_rest = self.v_rest
        v_reset = self.v_reset
        v_threshold = self.v_threshold
        decay_mult = self.decay_mult
        decay_shift = self.decay_shift
        refrac_steps = self.refrac_steps
        refrac_count = self._refrac_count
        if not (
            current.__class__ is int
            and v.__class__ is int
            and v_rest.__class__ is int
            and v_reset.__class__ is int
            and v_threshold.__class__ is int
            and decay_mult.__class__ is int
            and decay_shift.__class__ is int
            and refrac_steps.__class__ is int
            and refrac_count.__class__ is int
            and decay_mult >= 0
            and decay_shift >= 0
            and refrac_steps >= 0
            and refrac_count >= 0
        ):
            current = _require_int(current, "current")
            self._validate_configuration()
            v = self.v
            v_rest = self.v_rest
            v_reset = self.v_reset
            v_threshold = self.v_threshold
            decay_mult = self.decay_mult
            decay_shift = self.decay_shift
            refrac_steps = self.refrac_steps
            refrac_count = self._refrac_count
        if refrac_count > 0:
            self._refrac_count = refrac_count - 1
            return 0
        next_v = ((v - v_rest) * decay_mult >> decay_shift) + v_rest + current
        if next_v >= v_threshold:
            self.v = v_reset
            self._refrac_count = refrac_steps
            return 1
        self.v = next_v
        return 0

    def reset(self) -> None:
        """Restore ``v`` to ``v_rest`` and clear the refractory counter.

        Raises
        ------
        ValueError
            If ``v_rest`` is non-finite.
        TypeError
            If ``v_rest`` is not an integer.
        """
        v_rest = _require_int(self.v_rest, "v_rest")
        self.v = v_rest
        self._refrac_count = 0
