# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio routes for parameter fitting

"""Fit a model's parameters to recordings, and replay an exported fit.

The model is a catalogue model's canonical schema or a Universal DSL model sent
with the request (a candidate's). The request states the domains, the fixed
parameters and a cohort already split into training and hold-out recordings;
see :mod:`sc_neurocore.fitting`. A fit runs synchronously, so its size is
bounded before it starts: the number of model steps the optimiser can take is
estimated from the domains, the population, the generations and the training
recordings, and a larger fit is refused with that estimate rather than cut
short.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.api.runtime import StudioApiContext

MAX_SYNC_FIT_STEPS = 3_000_000
"""Most model steps a synchronous fit may take, estimated before it starts."""


class DomainBody(BaseModel):
    """One fitted parameter's search domain."""

    model_config = ConfigDict(extra="forbid")

    name: str
    low: float
    high: float
    scale: Literal["linear", "log"] = "linear"


class RecordingBody(BaseModel):
    """One stimulus and the observed response."""

    model_config = ConfigDict(extra="forbid")

    name: str
    current: list[float]
    observed: list[float]


class FitRequest(BaseModel):
    """A fit: the model, what to fit, and the split cohort."""

    model_config = ConfigDict(extra="forbid")

    catalogue_model: str | None = None
    schema_document: dict[str, Any] | None = Field(default=None, alias="schema")
    observable: str
    domains: list[DomainBody]
    fixed: dict[str, float] = Field(default_factory=dict)
    train: list[RecordingBody]
    holdout: list[RecordingBody]
    seed: int = 0
    generations: int = Field(default=40, ge=1, le=500)
    population: int = Field(default=12, ge=4, le=100)


class ReplayRequest(BaseModel):
    """An exported fit result."""

    model_config = ConfigDict(extra="forbid")

    result: dict[str, Any]


def estimated_fit_steps(request: FitRequest) -> int:
    """Upper estimate of the model steps a fit takes.

    Differential evolution evaluates ``population x parameters`` members per
    generation, plus the initial population; the local polish and the
    identifiability differences add a few evaluations per parameter, counted
    here as one more generation.
    """
    members = request.population * max(1, len(request.domains))
    evaluations = members * (request.generations + 2)
    return evaluations * sum(len(recording.current) for recording in request.train)


def _schema(request: FitRequest) -> dict[str, Any]:
    if (request.catalogue_model is None) == (request.schema_document is None):
        raise ValueError("name a catalogue model or send a schema, not both and not neither")
    if request.schema_document is not None:
        return request.schema_document
    from sc_neurocore.neurons.model_identity import ModelIdentityError, schema_for_class
    from sc_neurocore.neurons.universal_dsl import load_schema

    try:
        return load_schema(schema_for_class(str(request.catalogue_model)))
    except (ModelIdentityError, FileNotFoundError) as exc:
        raise ValueError(f"{request.catalogue_model} has no canonical schema to fit") from exc


def _refuse(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail={"reason": "invalid_fit", "message": message})


def build_fits_router(context: StudioApiContext) -> APIRouter:
    """Build the parameter-fitting router.

    Parameters
    ----------
    context:
        Shared runtime state; the fitting routes hold none of their own.
    """
    del context
    router = APIRouter()

    @router.post("/api/fits")
    def api_fit(request: FitRequest) -> dict[str, Any]:
        """Fit the model to the training recordings and validate on the hold-out ones."""
        from sc_neurocore.fitting import FitProblem, ParameterDomain, Recording, fit_parameters

        steps = estimated_fit_steps(request)
        if steps > MAX_SYNC_FIT_STEPS:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "fit_too_large",
                    "message": (
                        f"the fit may take {steps} model steps; a synchronous fit takes at most "
                        f"{MAX_SYNC_FIT_STEPS}. Fewer generations, a smaller population or "
                        "shorter recordings bring it within the bound."
                    ),
                    "estimated_steps": steps,
                    "max_steps": MAX_SYNC_FIT_STEPS,
                },
            )
        try:
            problem = FitProblem(
                schema=_schema(request),
                observable=request.observable,
                domains=tuple(
                    ParameterDomain(domain.name, domain.low, domain.high, domain.scale)
                    for domain in request.domains
                ),
                fixed=dict(request.fixed),
                train=tuple(
                    Recording(item.name, tuple(item.current), tuple(item.observed))
                    for item in request.train
                ),
                holdout=tuple(
                    Recording(item.name, tuple(item.current), tuple(item.observed))
                    for item in request.holdout
                ),
                seed=request.seed,
            )
            return fit_parameters(
                problem, generations=request.generations, population=request.population
            )
        except ValueError as exc:
            raise _refuse(str(exc)) from exc

    @router.post("/api/fits/replay")
    def api_fit_replay(request: ReplayRequest) -> dict[str, Any]:
        """Run an exported fit again and say whether it reproduced."""
        from sc_neurocore.fitting import replay_fit

        try:
            return replay_fit(request.result)
        except (KeyError, TypeError, ValueError) as exc:
            raise _refuse(f"the result cannot be replayed: {exc}") from exc

    return router


__all__ = [
    "FitRequest",
    "MAX_SYNC_FIT_STEPS",
    "ReplayRequest",
    "build_fits_router",
    "estimated_fit_steps",
]
