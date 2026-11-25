# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BrainChip Akida 2021 — event-domain rank-order IF neuron

"""BrainChip Akida event-domain rank-order integrate-and-fire neuron, in integer arithmetic."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass


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


def _require_unit_float(value: object, name: str) -> float:
    """Validate that ``value`` is a finite real scalar in ``[0, 1]`` and return ``float``."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number, not bool")
    if isinstance(value, numbers.Real):
        coerced = float(value)
        if not math.isfinite(coerced):
            raise ValueError(f"{name} must be finite")
        if coerced < 0.0 or coerced > 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
        return coerced
    raise TypeError(f"{name} must be a finite real number")


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
class AkidaNeuron:
    """BrainChip Akida 2021 — event-domain rank-order IF neuron.

    Membrane integrates weighted spikes with rank-order decay:
    V += int(weight * modulation^rank)
    Spike when V >= threshold. No leak between events.

    A weight of 0 is no event, so the rank counts the nonzero events. The
    factor modulation^rank is kept as a running product. The neuron fires
    at most once until reset, and V goes on integrating after the spike.
    The engine's AkidaNeuron follows the same rule.

    Reference: BrainChip Inc. (2021). Akida Neuromorphic Processor Reference Manual.
    """

    v: int = 0
    threshold: int = 100
    modulation: float = 0.75
    _rank: int = 0
    _spiked: bool = False
    _current_modulation: float = 1.0

    def __post_init__(self) -> None:
        """Validate accumulator, threshold, rank-order decay and internal state."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name in ("v", "threshold"):
            value: object = getattr(self, name)
            _validate_finite_real(value, name)
            if type(value) is not int and isinstance(value, numbers.Integral):
                setattr(self, name, int(value))
        self.modulation = _require_unit_float(self.modulation, "modulation")
        self._current_modulation = _require_unit_float(
            self._current_modulation, "_current_modulation"
        )
        self._rank = _require_int(self._rank, "_rank")
        if self._rank < 0:
            raise ValueError("_rank must be non-negative")
        if type(self._spiked) is not bool:
            raise TypeError("_spiked must be a bool")

    def step(self, weight: int) -> int:
        """Process one input spike event with given synaptic weight.

        Parameters
        ----------
        weight : int
            Synaptic weight of the incoming event; 0 is treated as no event.
            Finite real floats are accepted for network drivers and truncated
            to integer after rank-order scaling.

        Returns
        -------
        int
            1 on the first step where ``v`` reaches or exceeds ``threshold``
            since the last reset, else 0.

        Raises
        ------
        ValueError
            If ``weight`` or any field is non-finite, ``modulation`` or
            ``_current_modulation`` lies outside ``[0, 1]``, or ``_rank`` is
            negative. State is left unchanged.
        TypeError
            If ``weight`` or any numeric field is a boolean or non-real value,
            ``_rank`` is not an integer, or ``_spiked`` is not a ``bool``.
            State is left unchanged.
        FloatingPointError
            If scaling ``weight`` by ``_current_modulation`` overflows
            floating-point range. State is left unchanged.
        """
        v = self.v
        threshold = self.threshold
        modulation = self.modulation
        rank = self._rank
        spiked = self._spiked
        cur_mod = self._current_modulation
        if not (
            weight.__class__ is int
            and v.__class__ is int
            and threshold.__class__ is int
            and modulation.__class__ is float
            and rank.__class__ is int
            and spiked.__class__ is bool
            and cur_mod.__class__ is float
            and 0.0 <= modulation <= 1.0
            and 0.0 <= cur_mod <= 1.0
            and rank >= 0
        ):
            _validate_finite_real(weight, "weight")
            if isinstance(weight, numbers.Integral):
                weight = int(weight)
            self._validate_configuration()
            v = self.v
            threshold = self.threshold
            modulation = self.modulation
            rank = self._rank
            spiked = self._spiked
            cur_mod = self._current_modulation

        if weight != 0:
            # Iterative multiplication avoids repeated power evaluation on hot paths.
            try:
                product = weight * cur_mod
            except OverflowError as exc:
                raise FloatingPointError(
                    "Akida scaled weight overflowed floating-point range"
                ) from exc
            v = v + int(product)
            self.v = v
            self._rank = rank + 1
            self._current_modulation = cur_mod * modulation

        if v >= threshold and not spiked:
            self._spiked = True
            return 1
        return 0

    def reset(self) -> None:
        """Clear the accumulator, the rank, the weight factor and the spiked flag; the parameters stay."""
        self.v = 0
        self._rank = 0
        self._spiked = False
        self._current_modulation = 1.0
