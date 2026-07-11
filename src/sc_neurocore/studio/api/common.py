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
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException


logger = logging.getLogger("sc_neurocore.studio.app")


def _safe(fn: Callable[..., Any]) -> Any:
    try:
        return fn()
    except HTTPException:
        raise
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=422, detail="Invalid input") from None
    except Exception:
        logger.exception("Studio API internal error in %r", fn)
        raise HTTPException(status_code=500, detail="Internal error") from None
