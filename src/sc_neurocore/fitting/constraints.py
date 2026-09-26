# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parameter constraints in physical value space

"""Declare bounded linear combinations of named parameter values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterConstraint:
    """Require ``low <= sum(coefficients[name] * value[name]) <= high``.

    Coefficients and bounds are in the model's value space, including when a
    parameter is searched logarithmically. Bounds must define a nonempty interval; equality constraints are refused.
    """

    name: str
    coefficients: Mapping[str, float]
    low: float
    high: float

    def __post_init__(self) -> None:
        """Refuse empty, nonfinite or reversed constraint definitions."""
        if not self.name.strip() or not self.coefficients:
            raise ValueError("a constraint needs a name and coefficients")
        if not all(math.isfinite(value) for value in self.coefficients.values()):
            raise ValueError("constraint coefficients must be finite")
        if not any(self.coefficients.values()):
            raise ValueError("a constraint needs at least one nonzero coefficient")
        if not (math.isfinite(self.low) and math.isfinite(self.high)) or self.low >= self.high:
            raise ValueError("constraint bounds must be finite with low < high")

    def value(self, parameters: Mapping[str, float]) -> float:
        """Evaluate this combination without changing parameter units."""
        return sum(
            coefficient * parameters[name] for name, coefficient in self.coefficients.items()
        )

    def accepts(self, parameters: Mapping[str, float]) -> bool:
        """Return whether the value obeys both bounds without clamping."""
        return self.low <= self.value(parameters) <= self.high

    def to_public_dict(self) -> dict[str, Any]:
        """Export the whole named constraint for deterministic replay."""
        return {
            "name": self.name,
            "coefficients": dict(self.coefficients),
            "low": self.low,
            "high": self.high,
        }


def constraints_from_dict(entries: list[dict[str, Any]]) -> tuple[ParameterConstraint, ...]:
    """Read exported constraints, preserving all coefficients and bounds."""
    for entry in entries:
        if set(entry) != {"name", "coefficients", "low", "high"}:
            raise ValueError("constraint documents have missing or unknown fields")
        if not isinstance(entry["name"], str) or not isinstance(entry["coefficients"], Mapping):
            raise ValueError("constraints need a string name and named numeric coefficients")
        values = [entry["low"], entry["high"], *entry["coefficients"].values()]
        if any(type(value) not in (int, float) for value in values):
            raise ValueError("constraint coefficients and bounds must be JSON numbers")
    return tuple(
        ParameterConstraint(
            name=str(row["name"]),
            coefficients={str(k): float(v) for k, v in row["coefficients"].items()},
            low=float(row["low"]),
            high=float(row["high"]),
        )
        for row in entries
    )
