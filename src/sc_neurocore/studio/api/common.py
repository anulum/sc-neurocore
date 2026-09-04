# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared Studio API error boundary

"""Translate route-adapter failures without exposing internal details."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from sc_neurocore.studio.model_run_contract import ModelInputError, ModelSimulationFailure


logger = logging.getLogger("sc_neurocore.studio.app")


def _safe(fn: Callable[..., Any]) -> Any:
    """Run ``fn`` and translate failures into the public HTTP error contract.

    Model-run contract errors (:class:`ModelInputError`,
    :class:`ModelSimulationFailure`) become HTTP 422 with their structured
    public detail; other input errors become a generic 422 and anything else a
    logged 500 without internal details.
    """
    try:
        return fn()
    except HTTPException:
        raise
    except (ModelInputError, ModelSimulationFailure) as exc:
        raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=422, detail="Invalid input") from None
    except Exception:
        logger.exception("Studio API internal error in %r", fn)
        raise HTTPException(status_code=500, detail="Internal error") from None


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats by their text form so the error body stays JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


async def request_validation_error_handler(request: Request, exc: Exception) -> Response:
    """Return HTTP 422 for a body-validation failure even when the body carried NaN or Inf.

    The default handler echoes the offending input inside the error detail; a
    non-finite float there makes the JSON encoder raise and turns a rejected
    request into HTTP 500. This handler renders such inputs as text instead.
    """
    if not isinstance(exc, RequestValidationError):
        raise exc
    detail = _json_safe(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": detail})
