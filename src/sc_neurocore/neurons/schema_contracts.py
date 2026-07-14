# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared schema semantic contracts

"""Shared semantic predicates for schema validation and execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

StatelessEventKind = Literal["level_threshold", "poisson"]


def stateless_event_kind(schema: Mapping[str, object]) -> StatelessEventKind | None:
    """Return the supported event-only contract declared by ``schema``.

    A schema is event-only only when both its state and dynamics tables are
    present and empty. Deterministic schemas must provide a non-empty level
    threshold; stochastic Poisson schemas must provide a non-empty probability
    expression. Free-form extension labels are deliberately ignored.

    Parameters
    ----------
    schema:
        Parsed schema mapping.

    Returns
    -------
    StatelessEventKind or None
        The exact supported event contract, or ``None`` when the schema must
        declare ordinary state and dynamics.
    """

    state = schema.get("state")
    dynamics = schema.get("dynamics")
    threshold = schema.get("threshold")
    if (
        not isinstance(state, Mapping)
        or state
        or not isinstance(dynamics, Mapping)
        or dynamics
        or not isinstance(threshold, Mapping)
    ):
        return None

    detection = threshold.get("detection", "level")
    condition = threshold.get("condition")
    probability_expression = threshold.get("probability_expression")
    if (
        detection == "poisson"
        and isinstance(probability_expression, str)
        and probability_expression.strip()
    ):
        return "poisson"
    if detection == "level" and isinstance(condition, str) and condition.strip():
        return "level_threshold"
    return None


__all__ = ["StatelessEventKind", "stateless_event_kind"]
