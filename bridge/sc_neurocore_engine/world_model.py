# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge wrapper for world-model engine functions

"""Stable bridge wrappers for world-model Rust engine entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["get_lgssm_kalman_filter"]


def get_lgssm_kalman_filter() -> Callable[..., Any]:
    """Return the Rust LGSSM Kalman filter or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_lgssm_kalman_filter

    return py_lgssm_kalman_filter
