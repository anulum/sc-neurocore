# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Interval arithmetic checker for stochastic probability

"""Interval arithmetic checks for stochastic probability bounds and energy budgets.

An :class:`Interval` has finite ends in order. :class:`FormalVerifier` checks two properties by interval arithmetic,
not by an SMT solver, and says ``HELD`` only for inputs inside each property's domain: probability intervals inside
``[0, 1]``, and an energy and cost that are finite and non-negative.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Interval:
    """A closed interval ``[min_val, max_val]`` with finite ends and ``min_val <= max_val``.

    Raises
    ------
    ValueError
        If an end is NaN or infinite (an arithmetic overflow included), or the ends are reversed.
    """

    min_val: float
    max_val: float

    def __post_init__(self) -> None:
        """Refuse an interval whose ends are not finite or not in order."""
        if not (math.isfinite(self.min_val) and math.isfinite(self.max_val)):
            raise ValueError(f"interval ends must be finite, got [{self.min_val}, {self.max_val}]")
        if self.min_val > self.max_val:
            raise ValueError(f"interval ends are reversed: [{self.min_val}, {self.max_val}]")

    def __add__(self, other: Interval) -> Interval:
        """Return the interval of every sum of a point of each operand."""
        return Interval(self.min_val + other.min_val, self.max_val + other.max_val)

    def __mul__(self, other: Interval) -> Interval:
        """Return the interval of every product of a point of each operand."""
        vals = [
            self.min_val * other.min_val,
            self.min_val * other.max_val,
            self.max_val * other.min_val,
            self.max_val * other.max_val,
        ]
        return Interval(min(vals), max(vals))

    def __repr__(self) -> str:
        """Return the interval with four decimals per end."""
        return f"[{self.min_val:.4f}, {self.max_val:.4f}]"


def _within_unit(interval: Interval) -> bool:
    """Return whether ``interval`` lies inside ``[0, 1]``, the range of a probability."""
    return interval.min_val >= 0.0 and interval.max_val <= 1.0


class FormalVerifier:
    """Interval arithmetic checker for stochastic probability bounds and energy safety constraints.

    Not an SMT solver. Each check returns ``True`` (``HELD``) only when its inputs lie in the property's domain and
    the property holds for every point of them.
    """

    @staticmethod
    def verify_probability_bounds(input_interval: Interval, weight_interval: Interval) -> bool:
        """Show that an AND-gate output probability stays in ``[0, 1]``.

        The output is ``P(A and B) = P(A) * P(B)`` for independent streams. Both operands must be probability
        intervals; a negative or above-one operand shows nothing, even when the product lands in ``[0, 1]``.

        Parameters
        ----------
        input_interval, weight_interval : Interval
            The input and weight probability intervals.

        Returns
        -------
        bool
            ``True`` when both operands lie in ``[0, 1]`` and so does their product.
        """
        for name, interval in (("input", input_interval), ("weight", weight_interval)):
            if not _within_unit(interval):
                logger.info(
                    "Property (0 <= p <= 1): NOT SHOWN; %s %s is not a probability interval",
                    name,
                    interval,
                )
                return False
        out = input_interval * weight_interval

        is_safe = _within_unit(out)
        logger.info(
            "Verification: Input %s * Weight %s -> Output %s", input_interval, weight_interval, out
        )
        logger.info("Property (0 <= p <= 1): %s", "HELD" if is_safe else "VIOLATED")
        return is_safe

    @staticmethod
    def verify_energy_safety(energy: float, cost: float) -> bool:
        """Show that an operation does not consume more energy than is available.

        Parameters
        ----------
        energy : float
            Energy available; finite and non-negative.
        cost : float
            Energy the operation consumes; finite and non-negative.

        Returns
        -------
        bool
            ``True`` when both values are in their domain and ``energy - cost >= 0``.
        """
        for name, value in (("energy", energy), ("cost", cost)):
            if not (math.isfinite(value) and value >= 0.0):
                logger.warning(
                    "Verification: %s %s must be finite and non-negative. NOT SHOWN (Halt).",
                    name,
                    value,
                )
                return False
        if energy >= cost:
            new_e = energy - cost
            logger.info("Verification: %s - %s = %s >= 0. HELD.", energy, cost, new_e)
            return True
        logger.warning("Verification: %s < %s. VIOLATED (Halt).", energy, cost)
        return False
