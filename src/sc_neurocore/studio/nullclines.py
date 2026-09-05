# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Nullclines of a two-variable drift field with validity masks

"""Nullclines of an equation system on a grid, with per-component validity.

A nullcline point is a sign change of one component of the drift field
between adjacent grid samples. Where a component cannot be evaluated (a
domain error, an overflow, a non-finite value) the sample is *invalid* and no
cell touching it can carry a contour; the invalid part of the grid is
reported as a validity mask and summarised in the metric contract's domain
verdict. An invalid sample is never a zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.studio.analysis_contract import (
    MODEL_DEFINED_UNIT,
    DomainStatus,
    MetricContract,
    attach_contract,
)
from sc_neurocore.studio.model_run_contract import ModelInputError
from sc_neurocore.studio.simulation import ODE_MODEL_NAME

NULLCLINE_SCHEMA_VERSION = "studio.nullclines.v2"
DEFAULT_RANGES: dict[int, tuple[float, float]] = {0: (-80.0, 40.0), 1: (-1.0, 1.0)}

_EVALUATION_ERRORS: tuple[type[BaseException], ...] = (ArithmeticError, ValueError, TypeError)


def _input_error(field: str, reason: str) -> ModelInputError:
    return ModelInputError(model=ODE_MODEL_NAME, field=field, reason=reason)


def _build_neuron(
    equations: Sequence[str],
    params: Mapping[str, float],
    held: Mapping[str, float],
) -> EquationNeuron:
    try:
        return from_equations(*equations, params=dict(params), init=dict(held) or None, dt=0.01)
    except (ValueError, TypeError, SyntaxError, KeyError) as exc:
        raise _input_error("equations", str(exc)[:300]) from exc


def _sign_change(values: Sequence[float]) -> bool:
    return min(values) <= 0.0 <= max(values)


def nullclines_2d(
    equations: Sequence[str],
    params: Mapping[str, float],
    var_names: Sequence[str],
    ranges: Mapping[str, tuple[float, float]],
    grid_size: int = 80,
    *,
    current: float = 0.0,
    held: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute both nullclines of a two-variable section of the drift field.

    Parameters
    ----------
    equations:
        Equation strings of the system (``dx/dt = f(x, …)`` or map updates).
    params:
        Parameter values.
    var_names:
        The two swept variables ``(x, y)``; both must be equation variables.
    ranges:
        Inclusive ``(low, high)`` range per swept variable; a missing range
        falls back to :data:`DEFAULT_RANGES`.
    grid_size:
        Samples per axis.
    current:
        Input current ``I`` the field is evaluated at (held constant).
    held:
        Values at which every other equation variable is held; an
        unlisted variable is held at its equation-builder default (``0``).

    Returns
    -------
    dict
        ``var_names``, ``nullcline_0`` / ``nullcline_1`` (contour cell
        corners and the contour cell count), the grid axes, one validity
        mask per component (``1`` valid, ``0`` invalid; rows follow ``y``,
        columns ``x``), the held variables and input, and the
        :class:`~sc_neurocore.studio.analysis_contract.MetricContract`.

    Raises
    ------
    ModelInputError
        When fewer than two variables are given, a swept variable is not an
        equation variable, a held name is not an equation variable, a range
        is degenerate, or an expression references an unknown symbol.
    """
    if len(var_names) < 2:
        raise _input_error("var_names", "nullclines need two swept variables")
    if grid_size < 2:
        raise _input_error("grid_size", "grid_size must be at least 2")
    v0, v1 = str(var_names[0]), str(var_names[1])
    if v0 == v1:
        raise _input_error("var_names", "the two swept variables must differ")
    held_values = {str(name): float(value) for name, value in (held or {}).items()}
    neuron = _build_neuron(equations, params, held_values)
    declared = list(neuron.equations)
    for name in (v0, v1):
        if name not in declared:
            raise _input_error("var_names", f"{name!r} is not an equation variable")
    for name in held_values:
        if name not in declared:
            raise _input_error("held", f"{name!r} is not an equation variable")
        if name in (v0, v1):
            raise _input_error("held", f"{name!r} is swept and cannot be held")
    effective_held = {
        name: held_values.get(name, float(neuron.initial_state.get(name, 0.0)))
        for name in declared
        if name not in (v0, v1)
    }
    r0 = tuple(ranges.get(v0, DEFAULT_RANGES[0]))
    r1 = tuple(ranges.get(v1, DEFAULT_RANGES[1]))
    for name, bounds in ((v0, r0), (v1, r1)):
        if len(bounds) != 2 or not all(math.isfinite(float(b)) for b in bounds):
            raise _input_error("ranges", f"range of {name!r} must be two finite numbers")
        if float(bounds[0]) >= float(bounds[1]):
            raise _input_error("ranges", f"range of {name!r} must be increasing")

    x = np.linspace(float(r0[0]), float(r0[1]), grid_size)
    y = np.linspace(float(r1[0]), float(r1[1]), grid_size)
    X, Y = np.meshgrid(x, y)
    field = [np.full_like(X, np.nan), np.full_like(X, np.nan)]
    valid = [np.zeros(X.shape, dtype=bool), np.zeros(X.shape, dtype=bool)]
    compiled = [neuron._compiled_eqs[v0], neuron._compiled_eqs[v1]]

    base_env: dict[str, object] = dict(neuron._namespace)
    base_env.update(neuron.parameters)
    base_env.update(neuron.constants)
    base_env.update(effective_held)
    base_env["I"] = float(current)
    base_env["xi"] = 0.0

    with np.errstate(all="raise"):
        for i in range(grid_size):
            for j in range(grid_size):
                env = dict(base_env)
                env[v0] = float(X[i, j])
                env[v1] = float(Y[i, j])
                for component, code in enumerate(compiled):
                    try:
                        # Bandit B307 justification: the compiled expressions come from
                        # EquationNeuron._compiled_eqs, which already passed the AST
                        # whitelist of the equation safety validator; empty __builtins__
                        # blocks the residual escape vectors.
                        value = float(eval(code, {"__builtins__": {}}, env))  # nosec B307
                    except NameError as exc:
                        raise _input_error("equations", f"unknown symbol: {exc}") from exc
                    except _EVALUATION_ERRORS:
                        continue
                    if math.isfinite(value):
                        field[component][i, j] = value
                        valid[component][i, j] = True

    contours: list[list[list[float]]] = [[], []]
    for component in range(2):
        z = field[component]
        ok = valid[component]
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                if not (ok[i, j] and ok[i + 1, j] and ok[i, j + 1] and ok[i + 1, j + 1]):
                    continue
                corners = [z[i, j], z[i + 1, j], z[i, j + 1], z[i + 1, j + 1]]
                if _sign_change(corners):
                    contours[component].append([float(X[i, j]), float(Y[i, j])])

    cells_total = grid_size * grid_size
    valid_counts = [int(mask.sum()) for mask in valid]
    if all(count == cells_total for count in valid_counts):
        domain: DomainStatus = "complete"
    elif any(count == 0 for count in valid_counts):
        domain = "empty"
    else:
        domain = "partial"
    invalid_fraction = [1.0 - count / cells_total for count in valid_counts]

    contract = MetricContract(
        kind="nullclines",
        definition=(
            "sign change of one drift-field component across the four corners of a grid "
            "cell whose corners are all valid samples; the reported point is the cell's "
            "lower-left corner"
        ),
        units={v0: MODEL_DEFINED_UNIT, v1: MODEL_DEFINED_UNIT, "field": MODEL_DEFINED_UNIT},
        applicability=(
            "two-variable section of the system with every other variable held constant",
            f"input current held at I = {float(current)!r}",
            "drift field only: the diffusion-noise symbol xi is evaluated at 0",
            "differential equations integrated by the playground (explicit Euler); "
            "map updates are not a drift field and are not accepted here",
        ),
        limitations=(
            f"grid resolution {grid_size} samples per axis, no root refinement",
            "a sample where a component raises or is non-finite is invalid, not zero",
            "a cell with any invalid corner carries no contour",
        ),
        domain=domain,
        domain_detail={
            "cells_total": cells_total,
            "cells_valid": {v0: valid_counts[0], v1: valid_counts[1]},
            "invalid_fraction": {v0: invalid_fraction[0], v1: invalid_fraction[1]},
            "contour_cells": {v0: len(contours[0]), v1: len(contours[1])},
        },
    )
    payload: dict[str, Any] = {
        "schema_version": NULLCLINE_SCHEMA_VERSION,
        "var_names": [v0, v1],
        "nullcline_0": {"variable": v0, "points": contours[0], "cells": len(contours[0])},
        "nullcline_1": {"variable": v1, "points": contours[1], "cells": len(contours[1])},
        "grid": {"x": x.tolist(), "y": y.tolist(), "size": grid_size},
        "validity_0": valid[0].astype(np.int8).tolist(),
        "validity_1": valid[1].astype(np.int8).tolist(),
        "held": dict(effective_held),
        "current": float(current),
        "domain": {
            "status": domain,
            "invalid_fraction": {v0: invalid_fraction[0], v1: invalid_fraction[1]},
        },
    }
    return attach_contract(payload, contract)


__all__ = ["DEFAULT_RANGES", "NULLCLINE_SCHEMA_VERSION", "nullclines_2d"]
