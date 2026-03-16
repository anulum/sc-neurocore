# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Interval arithmetic checker for stochastic probability

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Interval:
    min_val: float
    max_val: float

    def __add__(self, other) -> None:
        return Interval(self.min_val + other.min_val, self.max_val + other.max_val)

    def __mul__(self, other) -> None:
        # Interval multiplication
        vals = [
            self.min_val * other.min_val,
            self.min_val * other.max_val,
            self.max_val * other.min_val,
            self.max_val * other.max_val,
        ]
        return Interval(min(vals), max(vals))

    def __repr__(self) -> None:
        return f"[{self.min_val:.4f}, {self.max_val:.4f}]"


class FormalVerifier:
    """
    Interval arithmetic checker for stochastic probability bounds and
    energy safety constraints. Not an SMT solver.
    """

    @staticmethod
    def verify_probability_bounds(input_interval: Interval, weight_interval: Interval) -> bool:
        """
        Prove that Output Probability is always in [0, 1].
        Logic: Out = Input * Weight (AND gate)
        """
        # Logic: P(A & B) = P(A) * P(B) assuming independence
        out = input_interval * weight_interval

        is_safe = out.min_val >= 0.0 and out.max_val <= 1.0
        logger.info(
            "Verification: Input %s * Weight %s -> Output %s", input_interval, weight_interval, out
        )
        logger.info("Property (0 <= p <= 1): %s", "HELD" if is_safe else "VIOLATED")
        return is_safe

    @staticmethod
    def verify_energy_safety(energy: float, cost: float) -> bool:
        """
        Prove that operation will not consume more energy than available.
        """
        # Symbolic check
        # Precondition: Energy >= Cost
        # Postcondition: NewEnergy >= 0
        if energy >= cost:
            new_e = energy - cost
            logger.info("Verification: %s - %s = %s >= 0. HELD.", energy, cost, new_e)
            return True
        else:
            logger.warning("Verification: %s < %s. VIOLATED (Halt).", energy, cost)
            return False
