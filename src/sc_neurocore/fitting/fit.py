# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fit parameters, validate on held-out data, state what is identifiable

"""Fit a model's parameters and say how far the fit can be trusted.

The objective is the mean squared residual of the observed variable over the
training recordings only. It is minimised by seeded differential evolution in
each parameter's search space (logarithmic where the domain says so), polished
locally; the best loss of every generation is kept as the optimiser's history,
and failed trials — a state that stopped being finite, or a mean squared residual
of 1e150 or more — are counted, not hidden. A fit that found no finite trial does
not report convergence.

The fitted parameters are then run on the hold-out recordings, which the
optimiser never saw, and their error is reported separately.

Identifiability and uncertainty come from the residual Jacobian at the
optimum, taken by central differences in search space. Its Gram matrix
``J^T J`` is decomposed; a direction whose eigenvalue is below
``IDENTIFIABILITY_RATIO`` of the largest is one the data do not constrain,
and it is reported with the parameter combination it moves. Standard errors
use the Gauss–Newton asymptotic covariance ``s^2 (J^T J)^-1`` with
``s^2 = RSS / (n - p)``, mapped back through the logarithm by the delta method
for log-scaled parameters. When any direction is unconstrained, no standard
error or correlation is reported: a covariance of a singular problem would be
certainty the data do not support.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.fitting.problem import (
    FIT_SCHEMA_VERSION,
    FitProblem,
    canonical_sha256,
    problem_from_dict,
    simulate,
)

UNCERTAINTY_METHOD = "gauss-newton-asymptotic"
IDENTIFIABILITY_RATIO = 1e-8
"""Smallest eigenvalue of ``J^T J``, relative to the largest, of a constrained direction."""

_FAILED_LOSS = 1e150
"""Loss of a failed trial. A trial fails when its state stops being finite or its mean
squared residual reaches this value; kept far below the float range so the optimiser's
own statistics over a population of failures stay finite."""


class _Objective:
    """The training loss, counting evaluations and failed trials."""

    def __init__(self, problem: FitProblem) -> None:
        self.problem = problem
        self.evaluations = 0
        self.failed = 0

    def parameters(self, internal: NDArray[np.float64]) -> dict[str, float]:
        values = {
            domain.name: domain.to_value(float(coordinate))
            for domain, coordinate in zip(self.problem.domains, internal, strict=True)
        }
        return {**self.problem.fixed, **values}

    def residuals(self, internal: NDArray[np.float64]) -> NDArray[np.float64] | None:
        parts: list[NDArray[np.float64]] = []
        for recording in self.problem.train:
            trace = simulate(
                self.problem.schema,
                self.problem.observable,
                self.parameters(internal),
                recording.current,
            )
            if trace is None:
                return None
            parts.append(trace - np.asarray(recording.observed, dtype=np.float64))
        return np.concatenate(parts)

    def __call__(self, internal: NDArray[np.float64]) -> float:
        self.evaluations += 1
        residual = self.residuals(internal)
        if residual is not None and np.all(np.isfinite(residual)):
            with np.errstate(over="ignore"):
                loss = float(np.mean(residual**2))
            # A finite trace can still be too large to square: that trial failed too.
            if loss < _FAILED_LOSS:
                return loss
        self.failed += 1
        return _FAILED_LOSS


def _rmse(
    problem: FitProblem, parameters: Mapping[str, float], recordings: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for recording in recordings:
        trace = simulate(problem.schema, problem.observable, parameters, recording.current)
        error: float | None = None
        if trace is not None:
            with np.errstate(over="ignore"):
                error = float(np.sqrt(np.mean((trace - np.asarray(recording.observed)) ** 2)))
        # A trace too large to square has no finite error; it is reported with the
        # diverged ones rather than as an infinity no JSON reader accepts.
        finite = error is not None and math.isfinite(error)
        rows.append(
            {"recording": recording.name, "rmse": error if finite else None, "diverged": not finite}
        )
    return rows


def _identifiability(
    objective: _Objective, optimum: NDArray[np.float64]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the identifiability diagnosis and the uncertainty at ``optimum``."""
    problem = objective.problem
    names = [domain.name for domain in problem.domains]
    base = objective.residuals(optimum)
    diverged = (
        {"identifiable": False, "reason": "the model diverged near the optimum"},
        {"method": UNCERTAINTY_METHOD, "standard_errors": None, "correlation": None},
    )
    if base is None:
        return diverged
    columns: list[NDArray[np.float64]] = []
    for index, domain in enumerate(problem.domains):
        low, high = domain.internal_bounds()
        step = 1e-4 * (high - low)
        forward, backward = optimum.copy(), optimum.copy()
        forward[index] += step
        backward[index] -= step
        plus, minus = objective.residuals(forward), objective.residuals(backward)
        if plus is None or minus is None:
            return diverged
        columns.append((plus - minus) / (2.0 * step))
    jacobian = np.stack(columns, axis=1)
    gram = jacobian.T @ jacobian
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = float(eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0
    unconstrained = [
        {
            "relative_eigenvalue": float(eigenvalues[k] / largest) if largest > 0 else 0.0,
            "direction": {
                name: float(component)
                for name, component in zip(names, eigenvectors[:, k], strict=True)
                if abs(component) >= 0.05
            },
        }
        for k in range(len(names))
        if largest <= 0 or eigenvalues[k] < IDENTIFIABILITY_RATIO * largest
    ]
    condition = math.inf if eigenvalues[0] <= 0 or largest <= 0 else largest / float(eigenvalues[0])
    diagnosis: dict[str, Any] = {
        "identifiable": not unconstrained,
        "condition_number": condition if math.isfinite(condition) else None,
        "eigenvalues": [float(value) for value in eigenvalues],
        "unconstrained_directions": unconstrained,
    }
    if unconstrained:
        return diagnosis, {
            "method": UNCERTAINTY_METHOD,
            "standard_errors": None,
            "correlation": None,
            "reason": "the data leave a parameter combination unconstrained",
        }
    dof = max(1, base.size - len(names))
    variance = float(np.sum(base**2)) / dof
    covariance = variance * np.linalg.inv(gram)
    internal_se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    values = objective.parameters(optimum)
    standard_errors = {
        domain.name: float(values[domain.name] * se if domain.scale == "log" else se)
        for domain, se in zip(problem.domains, internal_se, strict=True)
    }
    scale = np.outer(internal_se, internal_se)
    correlation = np.divide(covariance, scale, out=np.zeros_like(covariance), where=scale > 0)
    return diagnosis, {
        "method": UNCERTAINTY_METHOD,
        "residual_variance": variance,
        "standard_errors": standard_errors,
        "correlation": {
            left: {right: float(correlation[i, j]) for j, right in enumerate(names)}
            for i, left in enumerate(names)
        },
    }


def fit_parameters(
    problem: FitProblem, *, generations: int = 40, population: int = 12
) -> dict[str, Any]:
    """Fit ``problem`` and return the complete, replayable result.

    Parameters
    ----------
    problem:
        The model, domains and split cohort.
    generations, population:
        Differential-evolution size: at most ``generations`` generations of
        ``population`` members per fitted parameter.

    Returns
    -------
    dict
        The problem as given, the fitted values, training and hold-out error per
        recording, the optimiser history and trial counts, the identifiability
        diagnosis, the uncertainty, provenance, and the digest binding them.
    """
    import scipy
    from scipy.optimize import differential_evolution

    from sc_neurocore import __version__

    if generations < 1 or population < 4:
        raise ValueError("generations must be at least 1 and population at least 4")
    objective = _Objective(problem)
    history: list[dict[str, float | int]] = []

    def record(intermediate_result: Any) -> None:
        history.append({"generation": len(history) + 1, "loss": float(intermediate_result.fun)})

    outcome = differential_evolution(
        objective,
        [domain.internal_bounds() for domain in problem.domains],
        seed=problem.seed,
        maxiter=generations,
        popsize=population,
        tol=1e-10,
        polish=True,
        callback=record,
        updating="immediate",
        workers=1,
    )
    optimum = np.asarray(outcome.x, dtype=np.float64)
    parameters = objective.parameters(optimum)
    diagnosis, uncertainty = _identifiability(objective, optimum)
    body: dict[str, Any] = {
        "schema_version": FIT_SCHEMA_VERSION,
        "problem": problem.to_public_dict(),
        "fitted": {domain.name: parameters[domain.name] for domain in problem.domains},
        "training_loss": float(outcome.fun),
        "training": _rmse(problem, parameters, problem.train),
        "holdout": _rmse(problem, parameters, problem.holdout),
        "optimiser": {
            "method": "differential-evolution+polish",
            "generations_run": int(outcome.nit),
            "evaluations": objective.evaluations,
            "failed_trials": objective.failed,
            # Every member at the failure loss looks converged to the optimiser;
            # a fit that found no finite trial has not converged.
            "converged": bool(outcome.success) and float(outcome.fun) < _FAILED_LOSS,
            "message": str(outcome.message),
            "history": history,
        },
        "identifiability": diagnosis,
        "uncertainty": uncertainty,
        "provenance": {
            "schema_sha256": canonical_sha256(dict(problem.schema)),
            "train_sha256": [recording.data_sha256 for recording in problem.train],
            "holdout_sha256": [recording.data_sha256 for recording in problem.holdout],
            "seed": problem.seed,
            "generations": generations,
            "population": population,
            "sc_neurocore": __version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    body["result_sha256"] = canonical_sha256(body)
    return body


def replay_fit(result: Mapping[str, Any]) -> dict[str, Any]:
    """Run an exported fit again and say whether it reproduced.

    Returns
    -------
    dict
        ``reproduced`` is true when the new result's digest equals the exported
        one; both digests and the new result are included.
    """
    problem = problem_from_dict(result["problem"])
    provenance = result["provenance"]
    again = fit_parameters(
        problem,
        generations=int(provenance["generations"]),
        population=int(provenance["population"]),
    )
    return {
        "reproduced": again["result_sha256"] == result["result_sha256"],
        "exported_sha256": result["result_sha256"],
        "replayed_sha256": again["result_sha256"],
        "result": again,
    }


__all__ = [
    "IDENTIFIABILITY_RATIO",
    "UNCERTAINTY_METHOD",
    "fit_parameters",
    "replay_fit",
]
